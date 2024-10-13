from collections import defaultdict
from copy import deepcopy
import torch
import os

from transformers import DataCollatorForLanguageModeling
from datasets import Dataset, enable_progress_bar, disable_progress_bar

from ..constants import CODON_TABLE
from ._gen_utils import SEPStoppingCriteria
from tqdm import tqdm

import scanpy as sc
import numpy as np
import pandas as pd

from .configuration_decodon import DeCodonConfig
from .modeling_decodon import DeCodon
from .tokenization_decodon import DeCodonTokenizer

from typing import Optional, List, Tuple, Union


class AutoDeCodon:
    config_class = DeCodonConfig
    model_class = DeCodon
    tokenizer_class = DeCodonTokenizer

    PRETRAINED_WEIGHTS_MAP = {
        "goodarzilab/decodon-200M": "goodarzilab/decodon-200M",
        "goodarzilab/decodon-200M-euk": "goodarzilab/decodon-200M-euk",
    }

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = cls.config_class.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        model = cls.model_class.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            *model_args,
        )

        tokenizer = cls.tokenizer_class()
        tokenizer.model_max_length = model.config.max_position_embeddings

        taxids_filepath = os.path.join(os.path.dirname(__file__), "taxids.txt")
        with open(taxids_filepath, "r") as f:
            taxids = [f"<{tid.strip()}>" for tid in f.readlines()]

        tokenizer.set_organism_tokens(taxids)

        return cls(model, tokenizer)

    def get_embeddings(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        seq_col: str = "seq",
        taxid_col: Optional[str] = None,
        num_workers: Optional[int] = 1,
        bf16: Optional[bool] = False,
        layer_index: int = -1,
    ) -> sc.AnnData:
        """
        Extract Sequence Embeddings from the BERTrans model

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the sequences
        """
        num_layers = self.model.config.num_hidden_layers
        if layer_index == "all":
            layer_index = list(range(num_layers))
        elif isinstance(layer_index, int):
            layer_index = [layer_index % num_layers]
        elif isinstance(layer_index, list):
            layer_index = [idx % num_layers for idx in layer_index]
        else:
            raise ValueError("Invalid layer index")

        layer_index = sorted(layer_index)

        df["auto_seq"] = df[seq_col].apply(
            lambda x: f"{self.tokenizer.cls_token}{x}{self.tokenizer.sep_token}"
        )

        cols = ["auto_seq"] if taxid_col is None else ["auto_seq", taxid_col]

        dataset = Dataset.from_pandas(df[cols])

        def f(x):
            encoded = self.tokenizer(
                x["auto_seq"],
                truncation=True,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )

            if taxid_col is not None:
                encoded["taxid"] = torch.tensor(
                    [self.tokenizer.encoder[f"<{y}>"] for y in x[taxid_col]]
                )
                encoded["input_ids"][:, 0] = encoded["taxid"]

            return encoded

        dataset = dataset.map(
            lambda x: f(x),
            batched=True,
            num_proc=num_workers,
            remove_columns=["auto_seq"],
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            mlm_probability=0.0,
            pad_to_multiple_of=self.model.config.max_position_embeddings,
            return_tensors="pt",
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator.torch_call,
        )

        if bf16:
            self.model.to(torch.bfloat16)

        if torch.cuda.is_available():
            self.model.cuda()

        self.model.eval()

        embs = defaultdict(list)
        for batch in tqdm(data_loader):
            for key in batch:
                batch[key] = batch[key].to(self.model.device)

            with torch.no_grad():
                outputs = self.model.forward(
                    **batch, output_attentions=False, output_hidden_states=True
                )

            lens = (
                torch.argmax(
                    (batch["input_ids"] == self.tokenizer.sep_token_id).int(), dim=1
                )
                - 1
            )

            for layer_idx in layer_index:
                batch_embeddings = outputs.hidden_states[layer_idx]

                batch_embeddings = batch_embeddings[
                    torch.arange(batch_embeddings.shape[0]), lens
                ]

                if bf16:
                    batch_embeddings = batch_embeddings.float()

                embs[layer_idx].append(batch_embeddings.detach().cpu().numpy())

        X = np.concatenate(embs[layer_index[-1]])

        emb_adata = sc.AnnData(
            X=X,
        )
        emb_adata.obs = df.copy()

        for lidx in layer_index[:-1]:
            X = np.concatenate(embs[lidx])
            emb_adata.layers[f"layer_{lidx}"] = X

        if torch.cuda.is_available():
            self.model.cpu()

        return emb_adata

    def create_dataset_from_pandas(
        self,
        df: pd.DataFrame,
        seq_col: str = "seq",
        taxid_col: Optional[str] = None,
        position_col: Optional[str] = None,
        alt_codon_col: Optional[str] = None,
        verbose: bool = True,
        num_workers: int = 1,
    ):

        if verbose:
            enable_progress_bar()
        else:
            disable_progress_bar()

        cols = [seq_col]
        if position_col is not None:
            cols.append("pos")

        if alt_codon_col is not None:
            cols.append("alt_codon_encoded")

        if taxid_col is not None:
            cols.append(taxid_col)

        dataset = Dataset.from_pandas(df[cols])

        def f(x):
            encoded = self.tokenizer(
                [
                    f"{self.tokenizer.cls_token}{y}{self.tokenizer.sep_token}"
                    for y in x[seq_col]
                ],
                truncation=True,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )

            if taxid_col is not None:
                encoded["taxid"] = torch.tensor(
                    [self.tokenizer.encoder[f"<{y}>"] for y in x[taxid_col]]
                )
                encoded["input_ids"][:, 0] = encoded["taxid"]

            return encoded

        remove_cols = [seq_col]

        if taxid_col is not None:
            remove_cols.append(taxid_col)

        encoded = dataset.map(
            lambda x: f(x),
            batched=True,
            num_proc=num_workers,
            remove_columns=remove_cols,
        )
        return encoded

    def score_variants(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        seq_col: str = "seq",
        strand_col: str = "strand",
        position_col: str = "position",
        ref_allele_col: str = "ref_allele",
        alt_allele_col: str = "alt_allele",
        score_key_added: str = "codonbert_score",
        variant_mode: str = "allele",
        num_workers: int = 1,
        return_adata: bool = False,
        taxid_col: Optional[str] = None,
        bf16: Optional[bool] = True,
        **kwargs,
    ):
        """
        Score Variants using CodonBERT model

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing the variants. Must contain the following columns:
            - `seq_col`: Reference sequence
            - `position_col`: Position of the variant
            - `strand_col`: Strand of the variant

        """
        rev_comp = str.maketrans("ACGT", "TGCA")

        def prep_df(row):
            pos = row[position_col]
            codon_pos = pos // 3
            seq = row[seq_col]
            codons = [seq[i : i + 3] for i in range(0, len(seq), 3)]

            ref_allele = row[ref_allele_col]
            alt_allele = row[alt_allele_col]
            strand = row[strand_col]

            ref_codon = codons[codon_pos]

            if strand == "+":
                assert seq[pos] == ref_allele
                j = pos % 3
                alt_codon = ref_codon[:j] + alt_allele + ref_codon[j + 1 :]
            else:
                assert seq[pos] == ref_allele.translate(rev_comp)
                j = pos % 3
                alt_codon = (
                    ref_codon[:j] + alt_allele.translate(rev_comp) + ref_codon[j + 1 :]
                )

            row["ref_codon"] = ref_codon
            row["alt_codon"] = alt_codon
            row["var_codon_pos"] = codon_pos
            return row

        if variant_mode == "allele":
            df = df.apply(prep_df, axis=1)
        else:
            df = df.rename(
                {
                    ref_allele_col: "ref_codon",
                    alt_allele_col: "alt_codon",
                    position_col: "var_codon_pos",
                },
                axis=1,
                inplace=False,
            )

        df = df[df["ref_codon"].str.len() == 3]
        df = df[df["alt_codon"].str.len() == 3]
        df = df[df["ref_codon"] != df["alt_codon"]]

        def f(x):
            encoded = self.tokenizer(
                [
                    f"{self.tokenizer.cls_token}{y}{self.tokenizer.sep_token}"
                    for y in x[seq_col]
                ],
                truncation=True,
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            encoded["ref_codon"] = torch.tensor(
                [self.tokenizer.encoder[y] for y in x["ref_codon"]]
            )
            encoded["alt_codon"] = torch.tensor(
                [self.tokenizer.encoder[y] for y in x["alt_codon"]]
            )
            encoded["token_type_ids"] = torch.zeros_like(encoded["input_ids"])

            bs = encoded["input_ids"].shape[0]
            var_pos = torch.IntTensor(x["var_codon_pos"])
            encoded["position"] = var_pos + 1

            if taxid_col is not None:
                encoded["taxid"] = torch.tensor(
                    [self.tokenizer.encoder[f"<{y}>"] for y in x[taxid_col]]
                )

                encoded["input_ids"][:, 0] = encoded[
                    "taxid"
                ]  # set taxid as the CLS token

            return encoded

        if taxid_col is None:
            cols = [seq_col, "ref_codon", "alt_codon", "var_codon_pos"]
        else:
            cols = [seq_col, "ref_codon", "alt_codon", "var_codon_pos", taxid_col]

        dataset = Dataset.from_pandas(df[cols])

        if "__index_level_0__" in dataset.column_names:
            dataset = dataset.remove_columns(["__index_level_0__"])

        encoded = dataset.map(
            lambda x: f(x), batched=True, num_proc=num_workers, remove_columns=[seq_col]
        )

        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            mlm_probability=0.0,
            pad_to_multiple_of=self.model.config.max_position_embeddings,
            return_tensors="pt",
        )

        data_loader = torch.utils.data.DataLoader(
            encoded,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator.torch_call,
        )

        if bf16:
            self.model.to(torch.bfloat16)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.eval()
        scores = []
        seq_embeddings = []
        seq_var_embeddings = []
        for ref_batch in tqdm(data_loader):
            for k in ref_batch.keys():
                ref_batch[k] = ref_batch[k].to(self.model.device)

            alt_batch = deepcopy(ref_batch)
            alt_batch["input_ids"][
                torch.arange(ref_batch["input_ids"].shape[0]), ref_batch["position"]
            ] = ref_batch["alt_codon"]

            with torch.no_grad():
                ref_outputs = self.model.forward(**ref_batch, output_hidden_states=True)
                alt_outputs = self.model.forward(**alt_batch, output_hidden_states=True)

            ref_inputs = ref_batch["input_ids"]
            alt_inputs = alt_batch["input_ids"]

            bs, max_len = ref_inputs.shape

            ref_logits = ref_outputs.logits
            alt_logits = alt_outputs.logits

            ref_log_likelihoods = torch.log_softmax(
                ref_logits, dim=-1
            )  # (bs, max_len, vocab_size)
            alt_log_likelihoods = torch.log_softmax(
                alt_logits, dim=-1
            )  # (bs, max_len, vocab_size)

            ref_lls = (
                ref_log_likelihoods[:, :-1, :]
                .gather(-1, ref_inputs[:, 1:].unsqueeze(-1))
                .squeeze(-1)
            )  # (bs, max_len)
            ref_mask = (ref_inputs[:, 1:] != self.tokenizer.pad_token_id).float()
            ref_lls = ref_lls * ref_mask
            ref_lls = torch.sum(ref_lls, dim=-1) / torch.sum(ref_mask, dim=-1)  # (bs,)

            alt_lls = (
                alt_log_likelihoods[:, :-1, :]
                .gather(-1, alt_inputs[:, 1:].unsqueeze(-1))
                .squeeze(-1)
            )  # (bs, max_len)
            alt_mask = (alt_inputs[:, 1:] != self.tokenizer.pad_token_id).float()
            alt_lls = alt_lls * alt_mask
            alt_lls = torch.sum(alt_lls, dim=-1) / torch.sum(alt_mask, dim=-1)  # (bs,)

            batch_scores = alt_lls - ref_lls

            if bf16:
                batch_scores = batch_scores.float()

            scores.append(batch_scores.detach().cpu().numpy())

            if return_adata:
                lens = (
                    torch.argmax(
                        (ref_batch["input_ids"] == self.tokenizer.sep_token_id).int(),
                        dim=1,
                    )
                    - 1
                )

                ref_batch_embeddings = ref_outputs.hidden_states[-1][
                    torch.arange(bs), lens
                ]
                alt_batch_embeddings = alt_outputs.hidden_states[-1][
                    torch.arange(bs), lens
                ]

                batch_embeddings = alt_batch_embeddings - ref_batch_embeddings

                ref_var_embeddings = ref_outputs.hidden_states[-1][
                    torch.arange(bs), ref_batch["position"]
                ]
                alt_var_embeddings = alt_outputs.hidden_states[-1][
                    torch.arange(bs), ref_batch["position"]
                ]

                var_embeddings = ref_var_embeddings - alt_var_embeddings

                seq_embeddings.append(batch_embeddings.detach().cpu().numpy())
                seq_var_embeddings.append(var_embeddings.detach().cpu().numpy())

        scores = np.concatenate(scores)
        df[score_key_added] = scores
        df["syn_var"] = df[["ref_codon", "alt_codon"]].apply(
            lambda x: (
                "Synonymous"
                if CODON_TABLE.DNA[x[0]] == CODON_TABLE.DNA[x[1]]
                else "Non-Synonymous"
            ),
            axis=1,
        )

        if torch.cuda.is_available():
            self.model.cpu()

        if return_adata:
            seq_emb_adata = sc.AnnData(X=np.concatenate(seq_embeddings))
            seq_emb_adata.obs = df.copy()

            seq_emb_adata.layers["var_emb"] = np.concatenate(seq_var_embeddings)

            return seq_emb_adata
        else:
            return df

    @torch.no_grad()
    def generate(
        self,
        prompt: str = None,
        taxid: Optional[int] = 9606,  # human
        num_return_sequences: int = 1,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_k: int = 10,
        num_beams: int = 1,
        do_sample: bool = True,
        batch_size: int = 8,
        verbose: bool = False,
        bf16: Optional[bool] = True,
    ) -> List[str]:
        """
        Generates Coding sequences using the pretrained model. This method can be used in
        following ways:

        1. Generate CDS for a specific organism:
            ```python
            auto.generate(organism="human", num_return_sequences=10)
            ```

        2. Generate CDS for a specific organism with a prompt:
            ```python
            auto.generate(prompt="<human>ATGAAACGA", num_return_sequences=10)
            ```

        Parameters
        ----------
        prompt: Optional[str] (default=None)
            The prompt to use for generating the RNA transcripts.

        num_return_sequences: int (default=1)
            The number of RNA transcripts to generate.

        max_length: int (default=512)
            The maximum length of the generated RNA transcripts.

        temperature: float (default=1.0)
            The temperature of the generation process.

        top_k: int (default=10)
            The top-k tokens to sample from.

        num_beams: int (default=1)
            The number of beams to use during generation.

        do_sample: bool (default=True)
            Whether to use top-k sampling during generation or beam/greedy search.

        batch_size: int (default=8)
            The batch size to use during generation.

        repetition_penalty: float (default=1.2)
            The repetition penalty to use during generation.

        verbose: bool (default=False)
            Whether to print the generated sequences.

        Returns
        -------
        generated_seqs: List[str]
            A list of generated RNA transcripts.

        """
        if prompt is None:
            prompt = f"<{taxid}>"

        if bf16:
            self.model.to(torch.bfloat16)

        if torch.cuda.is_available():
            self.model.cuda()

        self.model.eval()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_special_tokens_mask=True,
            add_special_tokens=False,
        )

        self.model.config.eos_token_id = self.tokenizer.sep_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        batch_size = min(batch_size, num_return_sequences)

        gen_num_return_sequences = batch_size
        pbar = tqdm(
            range(num_return_sequences // batch_size),
            total=num_return_sequences // batch_size,
        )
        
        generated_seqs = []
        for _ in pbar:
            try:
                batch_gen_outputs = self.model.generate(
                    input_ids=inputs["input_ids"].to(self.model.device),
                    attention_mask=inputs["attention_mask"].to(self.model.device),
                    token_type_ids=inputs["token_type_ids"].to(self.model.device),
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    num_beams=num_beams,
                    num_return_sequences=gen_num_return_sequences,
                    do_sample=do_sample,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=[self.tokenizer.sep_token_id],
                    stopping_criteria=[
                        SEPStoppingCriteria(self.tokenizer, verbose=verbose)
                    ],
                    use_cache=False,  # TODO: need to fix the issue with KV cached inference
                )
            except KeyboardInterrupt:
                break

            batch_gen_seqs = self.tokenizer.batch_decode(
                batch_gen_outputs.sequences[:, 1:], skip_special_tokens=True
            )

            pbar.update(1)

            generated_seqs.extend(batch_gen_seqs)

        if torch.cuda.is_available():
            self.model = self.model.cpu()

        return generated_seqs