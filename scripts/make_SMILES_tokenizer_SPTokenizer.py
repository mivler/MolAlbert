"""
Trains and saves a SentencePiece Tokenizer for loading into a AlbertTokenizerFast.
Input is a datafile where each line is a SMILES string.

author: mivler
10/18/2023
"""
import tokenizers
import transformers
import os
import argparse


def main():
    # Parser Arguments
    parser = argparse.ArgumentParser(prog="SMILES_SPTokenizer", description="Takes in a file where each line is a SMILES string and trains a SentencePiece Tokenizer on it for loading into AlbertTokenizerFast")
    # Positional Args
    parser.add_argument("datafile", help="File used to train tokenizer, where each line is a SMILES string")
    # Optional Args
    parser.add_argument("-vs", "--vocab-size", default=2048, type=int, help="Size of vocabulary of tokenizer")
    parser.add_argument("-o", "--output-dir", default="models", help="Directory to save tokenizer in")

    args = parser.parse_args()

    # Read in all strings from data file and save as list
    with open(os.path.join(args.datafile), "r") as infile:
        all_train_smiles = [line.strip() for line in infile.readlines()]

    # Get max length of SMILES in data file
    max_len = int(max([len(smiles) for smiles in all_train_smiles]))
    # Remove SMILES iterator
    del all_train_smiles
    # Initialize special langauge tokens
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]

    # Initialize tokenizer
    tokenizer = tokenizers.SentencePieceBPETokenizer()
    # Train tokenizer on data file
    tokenizer.train(args.datafile, vocab_size=args.vocab_size, show_progress=True, special_tokens=special_tokens)

    # Creates AlbertTokenizerFast
    tokenizer_pretrained = transformers.AlbertTokenizerFast(tokenizer_object=tokenizer, model_max_length=max_len, special_tokens=special_tokens)

    # Defines special tokens for AlbertTokenizerFast object 
    tokenizer_pretrained.bos_token = "<S>"
    tokenizer_pretrained.pad_token = "[PAD]"
    tokenizer_pretrained.eos_token = "<T>"
    tokenizer_pretrained.unk_token = "[UNK]"
    tokenizer_pretrained.cls_token = "[CLS]"
    tokenizer_pretrained.sep_token = "[SEP]"
    tokenizer_pretrained.mask_token = "[MASK]"

    # Save pretrained tokenizer for use in Albert model training
    tokenizer_pretrained.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
