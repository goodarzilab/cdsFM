import torch

from transformers import DataCollatorForLanguageModeling
from datasets import Dataset, enable_progress_bar, disable_progress_bar
from .modeling_encodon import (
    EnCodon,
)
from .tokenization_encodon import EnCodonTokenizer
from tqdm import tqdm

import scanpy as sc
import numpy as np
import pandas as pd


from typing import Optional, List, Union

from ..constants import CODON_TABLE
from .configuration_encodon import (
    EnCodonConfig,
)


class AutoEnCodon:
    config_class = EnCodonConfig
    model_class = EnCodon
    tokenizer_class = EnCodonTokenizer

    PRETRAINED_WEIGHTS_MAP = {
        "goodarzilab/encodon-80M": "goodarzilab/encodon-80M",
        "goodarzilab/encodon-620M": "goodarzilab/encodon-620M",
        "goodarzilab/encodon-80M-euk": "goodarzilab/encodon-80M-euk",
        "goodarzilab/encodon-620M-euk": "goodarzilab/encodon-620M-euk",
    }

    def __init__(self, model: EnCodon, tokenizer: EnCodonTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = cls.config_class.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        model = cls.model_class.from_pretrained(
            pretrained_model_name_or_path, config=config, *model_args, **kwargs
        )

        tokenizer = cls.tokenizer_class()
        tokenizer.model_max_length = model.config.max_position_embeddings

        return cls(model, tokenizer)

    def create_dataset_from_pandas(
        self,
        df: pd.DataFrame,
        seq_col: str = "seq",
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

        dataset = Dataset.from_pandas(df[cols])
        dataset = dataset.map(
            lambda x: self.tokenizer(
                [
                    f"{self.tokenizer.cls_token}{y}{self.tokenizer.sep_token}"
                    for y in x[seq_col]
                ],
                add_special_tokens=False,
                return_attention_mask=True,
                return_special_tokens_mask=True,
                truncation=True,
            ),
            batched=True,
            num_proc=num_workers,
            remove_columns=[seq_col],
        )
        return dataset

    def get_embeddings(
        self,
        seqs: List[str],
        batch_size: int = 32,
        layer_index: int = -1,
        verbose: bool = True,
        bf16: Optional[bool] = False,
        num_workers: Optional[int] = 1,
        **kwargs,
    ) -> sc.AnnData:
        """
        Extract Sequence Embeddings from the BERTrans model

        Parameters
        ----------
        seqs : List[str]
            List of sequences
        batch_size : int
            Batch size for inference
        seq_col : str
            Column name containing the sequences in the DataFrame
        layer_index : int
            Layer index from which to extract the embeddings
        """
        if isinstance(seqs, str):
            seqs = [seqs]
            
        dataset = Dataset.from_pandas(pd.DataFrame({"seq": seqs}))
        dataset = dataset.map(lambda x: self.tokenizer(
            [f"{self.tokenizer.cls_token}{y}{self.tokenizer.sep_token}" for y in x["seq"]],
            add_special_tokens=False,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        ), batched=True, num_proc=num_workers, remove_columns=["seq"])
        
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

        num_layers = self.model.config.num_hidden_layers
        if isinstance(layer_index, int):
            layer_index = [layer_index % num_layers]
        elif isinstance(layer_index, list):
            layer_index = [l % num_layers for l in layer_index]
        elif layer_index == "all":
            layer_index = list(range(num_layers))
        else:
            raise ValueError(f"Invalid layer index: {layer_index}")

        layer_index = sorted(layer_index)

        embs = {layer_idx: [] for layer_idx in layer_index}

        if torch.cuda.is_available():
            self.model.cuda()

        if bf16:
            self.model.to(torch.bfloat16)

        self.model.eval()
        pbar = tqdm(data_loader) if verbose else data_loader
        for batch in pbar:
            for key in batch:
                batch[key] = batch[key].to(self.model.device)

            with torch.no_grad():
                outputs = self.model.forward(**batch, output_hidden_states=True)

            lens = batch["attention_mask"].sum(1) - 2  # excluding CLS and SEP tokens

            for lidx in layer_index:
                hidden_states = outputs.hidden_states[lidx]
                # Perform mean over non-special tokens of each sequence to extract sequence embeddings
                for hidden_state, seq_len in zip(hidden_states, lens):
                    seq_emb = hidden_state[1 : seq_len + 1].mean(dim=0)[
                        None, ...
                    ]  # (1, hidden_size)
                    if bf16:
                        seq_emb = seq_emb.float()
                    embs[lidx].append(seq_emb.detach().cpu().numpy())

        X = np.concatenate(embs[layer_index[-1]])  # (num_samples, hidden_size)

        emb_adata = sc.AnnData(
            X=X,
        )
        emb_adata.obs["seq"] = seqs

        for layer_idx in layer_index[:-1]:
            emb_adata.layers[f"layer_{layer_idx}"] = np.concatenate(embs[layer_idx])

        if torch.cuda.is_available():
            self.model.cpu()

        return emb_adata

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
        score_method: str = "wt_llr",
        num_workers: int = 1,
        bf16: bool = False,
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

        batch_size: int
            Batch size for evaluation

        seq_col: str
            Column name containing the sequences

        strand_col: str
            Column name containing the strand of the variant

        position_col: str
            Column name containing the position of the variant

        ref_allele_col: str
            Column name containing the reference allele

        alt_allele_col: str
            Column name containing the alternate allele

        score_key_added: str
            Key to be added to the DataFrame containing the scores

        score_method: str
            Method to calculate the score. Either "wt_llr" or "masked_llr"
            - "wt_llr": score = log(P(alt_codon|wt_seq)) - log(P(ref_codon|wt_seq))
            - "masked_llr": score = log(P(alt_codon|masked_seq)) - log(P(ref_codon|masked_seq))

        variant_mode: str
            Mode of the variant. Either "allele" or "codon"
        """
        rev_comp = str.maketrans("ACGT", "TGCA")

        def prep_df(row):
            pos = int(row[position_col])
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
            row["auto_var_codon_pos"] = codon_pos

            alt_codons = codons[:codon_pos] + [alt_codon] + codons[codon_pos + 1 :]
            alt_seq = "".join(alt_codons)

            row["alt_seq"] = alt_seq

            return row

        if variant_mode == "allele":
            df = df.apply(prep_df, axis=1)
        else:
            df = df.rename(
                {
                    ref_allele_col: "ref_codon",
                    alt_allele_col: "alt_codon",
                    position_col: "auto_var_codon_pos",
                },
                axis=1,
                inplace=False,
            )

            def get_alt_seq(row):
                pos = int(row["auto_var_codon_pos"])
                seq = row[seq_col]
                codons = [seq[i : i + 3] for i in range(0, len(seq), 3)]
                alt_codons = codons[:pos] + [row["alt_codon"]] + codons[pos + 1 :]
                return "".join(alt_codons)

            df["alt_seq"] = df.apply(get_alt_seq, axis=1)

        df = df[df["ref_codon"].str.len() == 3]
        df = df[df["alt_codon"].str.len() == 3]
        df = df[df["ref_codon"] != df["alt_codon"]]

        if score_method == "mut_llr":
            seq_col = "alt_seq"

        def f(x):
            encoded = self.tokenizer(
                [
                    f"{self.tokenizer.cls_token}{y}{self.tokenizer.sep_token}"
                    for y in x[seq_col]
                ],
                add_special_tokens=False,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded["ref_codon"] = torch.tensor(
                [self.tokenizer.encoder[y] for y in x["ref_codon"]]
            )
            encoded["alt_codon"] = torch.tensor(
                [self.tokenizer.encoder[y] for y in x["alt_codon"]]
            )

            # check if the variant position has the same codon as the reference codon
            var_pos = torch.IntTensor(x["auto_var_codon_pos"]) + 1

            if score_method == "mut_llr":
                assert torch.all(
                    encoded["input_ids"][torch.arange(len(x[seq_col])), var_pos]
                    == encoded["alt_codon"]
                ), f"Variant position does not have the same codon as the reference codon\n{encoded['input_ids'][:, var_pos].shape}\n{encoded['ref_codon'].shape}\n{var_pos}"
            else:
                assert torch.all(
                    encoded["input_ids"][torch.arange(len(x[seq_col])), var_pos]
                    == encoded["ref_codon"]
                ), f"Variant position does not have the same codon as the reference codon\n{encoded['input_ids'][:, var_pos].shape}\n{encoded['ref_codon'].shape}\n{var_pos}"

            bs = encoded["input_ids"].shape[0]
            if score_method == "masked_llr":
                encoded["input_ids"][
                    torch.arange(bs), var_pos
                ] = self.tokenizer.mask_token_id
            encoded["position"] = var_pos

            return encoded

        cols = [seq_col, "ref_codon", "alt_codon", "auto_var_codon_pos"]

        dataset = Dataset.from_pandas(df[cols])

        if "__index_level_0__" in dataset.column_names:
            dataset = dataset.remove_columns(["__index_level_0__"])

        encoded = dataset.map(
            lambda x: f(x),
            batched=True,
            num_proc=num_workers,
            remove_columns=[seq_col],
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
            self.model.cuda()

        self.model.eval()
        scores = []
        seq_embeddings = []
        seq_var_embeddings = []
        for batch in tqdm(data_loader):
            for k in batch.keys():
                batch[k] = batch[k].to(self.model.device)

            with torch.no_grad():
                outputs = self.model.forward(**batch, output_hidden_states=False)
                logits = outputs.logits

            inputs = batch["input_ids"]
            bs, max_len = inputs.shape

            log_likelihoods = torch.log_softmax(logits, dim=-1)

            lls = log_likelihoods[torch.arange(bs), batch["position"]]
            lls_ref = lls[torch.arange(bs), batch["ref_codon"]]
            lls_alt = lls[torch.arange(bs), batch["alt_codon"]]

            batch_scores = lls_alt - lls_ref

            if bf16:
                batch_scores = batch_scores.float()

            scores.append(batch_scores.detach().cpu().numpy())

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

        return df
