# Repository of Codon-based Foundation Models (EnCodon & DeCodon)
![cdsFM](./assets/sketch.png)
This repository contains the code for the EnCodon and DeCodon models, codon-resolution large language models pre-trained on the NCBI Genomes database described in the paper "[A Suite of Foundation Models Captures the Contextual Interplay Between Codons](https://www.biorxiv.org/content/10.1101/2024.10.10.617568v1)". 

## Get started 🚀
### Installation
#### From source
Currently, this is the only way to install the package but will push a pip installable version soon. To install the package from source, run the following command:

```bash
pip install git+https://github.com/goodarzilab/cdsFM.git
```

<!-- #### From pip

```bash
pip install cdsFM
``` -->

## Applications
Now that you have cdsFM installed, you can use `AutoEnCodon` and `AutoDeCodon` classes which serve as wrappers around the pre-trained models. Here are some examples on how to use them:

### Sequence Embedding Extraction with EnCodon
Following is an example of how to use the EnCodon model to extract sequence embeddings:

```python
from cdsFM import AutoEnCodon

# Load your dataframe containing sequences
seqs = ...

# Load a pre-trained EnCodon model
model = AutoEnCodon.from_pretrained("goodarzilab/encodon-620M")

# Extract embeddings
embeddings = model.get_embeddings(seqs, batch_size=32)
```

### Sequence Generation with DeCodon
You can generate organism-specific coding sequences with DeCodon simply by:

```python
from cdsFM import AutoDeCodon

# Load a pre-trained DeCodon model
model = AutoDeCodon.from_pretrained("goodarzilab/DeCodon-200M")

# Generate!
gen_seqs = model.generate(
    taxid=9606, # NCBI Taxonomy ID for Homo sapiens
    num_return_sequences=32, # Number of sequences to return
    max_length=1024, # Maximum length of the generated sequence
    batch_size=8, # Batch size for generation
)

```

--- 
## Tokenization

EnCodon and DeCodon are pre-trained on coding sequences of length up to 2048 codons (i.e. 6144 nucleotides), including the
\<CLS> token prepended automatically to the beginning of the sequence and the \<SEP> token appended at the end. The tokenizer's vocabulary consists of 64 codons and 5 special tokens namely \<CLS>, \<SEP>, \<PAD>, \<MASK> and \<UNK>. 

---

## HuggingFace 🤗

A collection of pre-trained checkpoints of EnCodon & DeCodon models are available on [HuggingFace 🤗](https://huggingface.co/goodarzilab). Following table contains the list of available models:

| Model | name | num. params | description | weights |
| :--- | :---: | :---: | :---: | :---: |
| EnCodon | encodon-80M | 80M | Pre-trained checkpoint | [🤗](https://huggingface.co/goodarzilab/EnCodon-80M) |
| EnCodon | encodon-80M-euk | 80M | Eukaryotic-expert | [🤗](https://huggingface.co/goodarzilab/EnCodon-80M-euk) |
| EnCodon | encodon-620M | 620M | Pre-trained checkpoint | [🤗](https://huggingface.co/goodarzilab/EnCodon-620M) |
| EnCodon | encodon-620M-euk | 620M | Eukaryotic-expert | [🤗](https://huggingface.co/goodarzilab/EnCodon-620M-euk) |
| DeCodon | decodon-200M | 200M | Pre-trained checkpoint | [🤗](https://huggingface.co/goodarzilab/DeCodon-200M) |
| DeCodon | decodon-200M-euk | 200M | Eukaryotic-expert | [🤗](https://huggingface.co/goodarzilab/DeCodon-200M-euk) |

---

## Citation

```bibtex
@article{Naghipourfar2024,
  title = {A Suite of Foundation Models Captures the Contextual Interplay Between Codons},
  url = {http://dx.doi.org/10.1101/2024.10.10.617568},
  DOI = {10.1101/2024.10.10.617568},
  publisher = {Cold Spring Harbor Laboratory},
  author = {Naghipourfar,  Mohsen and Chen,  Siyu and Howard,  Mathew and Macdonald,  Christian and Saberi,  Ali and Hagen,  Timo and Mofrad,  Mohammad and Coyote-Maestas,  Willow and Goodarzi,  Hani},
  year = {2024},
  month = oct 
}
```


