"""
Splits a zinc15 .csvfile into training, validation, and testing sets.

authors: mivler
10/18/2023
"""

import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(prog="zinc15_train_valid_test_split_smiles", description="Splits a sv from the zinc15 moleculeNet .csv file into training, validation, and testing sets.")
    # Positional Args
    parser.add_argument("infile")
    # Optional Args
    parser.add_argument("-r", "--random-state", type=int, default=42, help="Random state seed for data split")
    parser.add_argument("--train-perc", type=float, default=.75, help="Proportion of dataset to put into training set")
    parser.add_argument("--valid-perc", type=float, default=.1, help="Proportion of dataset to put into validation set")
    parser.add_argument("--test-perc", type=float, default=.15, help="Proportion of dataset to put into test set")
    parser.add_argument("-o", "--out-dir", default="data", help="Directory to output train, validation, and test files")

    args = parser.parse_args()

    # Read in csv infile
    df = pd.read_csv(args.infile)
    X = df["smiles"]

    # Set up proportions of validation and test sets
    test_size = args.test_perc
    valid_size = args.valid_perc / (args.valid_perc + args.train_perc)

    # Split dataset into test set and all other examples
    x_train_val, x_test = train_test_split(X, test_size=test_size, random_state=args.random_state)
    # Split remaining examples into train set and validation set
    x_train, x_val = train_test_split(x_train_val, test_size=valid_size, random_state=args.random_state)
    
    # Grab original filename of csv file (without file extension)
    original_filename = ".".join(args.infile.split(os.sep)[-1].split(".")[:-1])
    # Sets up outpath and file name
    out_name = os.path.join(args.out_dir, original_filename)

    # Saves each set split into a properly labeled file using the at location <out_dir>/<original_file_name>_<split_partition>.csv
    pd.Series(x_train).to_csv(os.path.join(f"{out_name}_train.csv"), index=False, header=False)
    pd.Series(x_val).to_csv(os.path.join(f"{out_name}_valid.csv"), index=False, header=False)
    pd.Series(x_test).to_csv(os.path.join(f"{out_name}_test.csv"), index=False, header=False)


if __name__ == "__main__":
    main()
