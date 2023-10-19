#!/bin/bash

# This program sets up the conda environment to run the scripts in this package.
# It then saves the enviroment as a .yml file
# author: mivler

env_name="molecular_nn_env"

conda create -n $env_name
conda activate $env_name

conda install python=3.8
conda install -c conda-forge deepchem
conda install -c conda-forge rdkit
conda install -c huggingface datasets
conda install -c huggingface transformers==4.14.1 tokenizers==0.10.3
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

conda env export > ${env_name}.yml
