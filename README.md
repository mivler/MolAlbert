# MolAlbert
This is MolAlbert, a set of scripts to train self-supervised ALBERT model to encode SMILES strings. This is a WIP being trained on the available ZINC15 dataset from MoleculeNet.

## Setup
First, you'll need to set up the conda environment. For this you will need conda. You can then either use the molecular\_nn\_env.yml or you can create the molecular\_nn\_env with the script in the 'scripts' directory. <br/>
In order to do so, just run: <br/>
`./scripts/create_molecular_nn_env.sh`
<br/>
<br/>
From there, we need to adjust some code within the conda environment. In order to do so, go to: <br/>
`<anaconda or miniconda path>/envs/molecular_nn_env/lib/python3.8/site-packages/transformers/models/albert/modeling_albert.py` <br/>

Once in this file, to go the class `AlbertForMaskedLM`. Adjust the arguments to the `self.albert` initialization by changing the argument `add_pooling_layer=False` to `add_pooling_layer=True`.
<br/> <br/>
Now you should be ready to go!

## Loading and Splitting SMILES Data
TODO
