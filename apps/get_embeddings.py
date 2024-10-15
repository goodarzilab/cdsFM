"""
    Extract Sequence Embeddings from a pre-trained EnCodon Model.
"""
import argparse
import os
import pandas as pd
from Bio import SeqIO
from cdsfm.encodon import AutoEnCodon
from collections import defaultdict
from tqdm import tqdm


def load_data(args):
    if args.input_file.endswith(".csv"):
        df = pd.read_csv(args.input_file)
        seqs = list(df[args.seq_col].values)
    elif args.input_file.endswith(".fasta") or args.input_file.endswith(".fa"):
        seqs = []
        for record in tqdm(SeqIO.parse(args.input_file, "fasta"), desc="Loading data"):
            seqs.append(str(record.seq))
    else:
        raise ValueError("Input file must be a csv or a fasta file")
    
    return seqs

def main(args):
    seqs = load_data(args)
    
    automodel = AutoEnCodon.from_pretrained(args.model_name)
    
    emb_adata = automodel.get_embeddings(
        seqs=seqs,
        batch_size=args.batch_size,
        layer_index=args.layer_index,
        num_workers=args.num_workers,
        verbose=args.verbose,
        bf16=args.bf16,
    )
    
    if args.verbose:
        print(emb_adata)
    
    emb_adata.write_h5ad(os.path.join(args.output_path, f"emb_{args.model_name.split('/')[-1]}.h5ad"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_file", '-i', type=str, required=True, help="Path to the input file (Can be a csv or a fasta file)")
    parser.add_argument("--model_name", '-m', type=str, default="encodon-80M", help="Name of the model to use or path to the model")
    parser.add_argument("--batch_size", '-b', type=int, default=32, help="Batch size")
    parser.add_argument("--output_path", '-o', type=str, required=True, help="Path to save the extracted embeddings")
    parser.add_argument("--num_workers", '-n', type=int, default=1, help="Number of workers to use for dataloading")
    parser.add_argument("--seq_col", '-sc', type=str, default="seq", help="Name of the column containing the sequences (Only used if input file is a csv)")
    parser.add_argument("--layer_index", '-l', type=int, default=-1, help="Layer index to extract embeddings from")
    parser.add_argument("--verbose", '-v', action="store_true", help="Print verbose output")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 for inference")
    
    args = parser.parse_args()
    main(args)
