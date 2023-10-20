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

    # Grab size and define outpath directory for data
    size = args.size
    outpath = os.path.join("data", f"zinc15_molnet_{size}")

    # Download zinc15 dataset of appropriate size
    featurizer = dc.feat.DummyFeaturizer()
    print(f"downloading to {outpath}")
    # NOTE: This will now work if there is any version of the zinc15 data in your /tmp directory 
    # This includes raw or processed data
    db = dc.molnet.load_zinc15(featurizer=featurizer, data_dir=outpath, dataset_size=size)


if __name__ == "__main__":
    main()
