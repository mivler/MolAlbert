"""
Downloads the zinc15 database

author: mivler
10/18/2023
"""

import deepchem as dc
import os
import sys
import argparse


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(prog="Downloads zinc15 database from deepchem MolNet")
    parser.add_argument("-s", "--size", default="250K")
    args = parser.parse_args()

    size = args.size

    # Download zinc15 dataset of appropriate size
    featurizer = dc.feat.DummyFeaturizer()
    db = dc.molnet.load_zinc15(featurizer=featurizer, data_dir=os.path.join("data", f"zinc15_molnet_{size}"), dataset_size=size)


if __name__ == "__main__":
    main()
