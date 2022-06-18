# LIMO: Latent Inceptionism for Targeted Molecule Generation

This is the repository for our 2022 ICML paper "LIMO: Latent Inceptionism for Targeted Molecule Generation" by Peter Eckmann, Kunyang Sun, Bo Zhao, Mudong Feng, Michael K. Gilson, and Rose Yu.


## Installation

Please ensure that [RDKit](https://www.rdkit.org/docs/Install.html) and [Open Babel](https://openbabel.org/wiki/Category:Installation) are installed. The following Python packages are also required (these can also be installed with `pip install -r requirements.txt`):

```
torch
pytorch-lightning
selfies
scipy
tqdm
```

Code was tested with Python 3.9, but will likely work on any version of Python 3.

## Generating molecules

Call `python generate_molecules.py` to generate molecules with desired properties. Run `--help` to see the full list of supported properties and other parameters. The default setting is to perform multi-objective binding affinity maximiziation with the filtering step, but other properties can be optimized by specifying `--prop` (see "Training the property predictor" to optimize your own properties). Model files for penalized logP, binding affinity to ESR1 and ACAA1, QED, and SA are provided in the `property_models` folder. The default binding affinity model is ESR1 (which is `binding_affinity.pt`), but to optimize binding to another protein one must make sure the `binding_affinity.pt` file contains the model for that correct protein. For example, for 2IIK binding optimization one must remove/rename the original `binding_affinity.pt` file and rename `2iik_binding_affinity.pt` to `binding_affinity.pt`. A trained VAE model is provided that was trained on the ZINC250K dataset, but any SMILES dataset can be used for training (see "Training the VAE").

To optimize molecules for binding affinity, an AutoDock-GPU executable [must be compiled](https://github.com/ccsb-scripps/AutoDock-GPU#compilation), and pointed to with the `--autodock_executable` flag when generating molecules, training the property predictor, or fine-tuning. A `--protein_file` must also be specified, but files for ESR1 (`1err/1err.maps.fld`) and ACAA1 (`2iik/2iik.maps.fld`) are already supplied. To generate your own protein files, see Steps 1-2 in the [AutoDock4 User Guide](https://autodock.scripps.edu/wp-content/uploads/sites/56/2021/10/AutoDock4.2.6_UserGuide.pdf). The [AutoDock-GPU Wiki](https://github.com/ccsb-scripps/AutoDock-GPU/wiki/Guideline-for-users) may also be helpful.

## Fine-tuning

Run `python fine-tune "<smiles>"` to fine-tune a molecule for increased binding affinity. For each iteration, it will print out the binding affinity and SMILES of the best molecule. Run with `2>/dev/null` to silence RDKit warnings.

## Training the property predictor 

Train the property predictor with `python train_property_predictor.py --prop {logp, penalized_logp, qed, sa, binding_affinity}`. Run with `--help` for a full list of options. When training for binding affinity, you must also provide a `--protein_file`, with the default being `1err/1err.maps.fld`. For properties other than binding affinity, we suggest training with `--num_mols 100000` for greater prediction accuracy. The printed `r` value should be >0.5 for most properties (except binding affinity, which is typically a lower value because it is a more difficult task), so running multiple times until you reach a suitable `r` value or training with more molecules is recommended.

## Training the VAE

Before training, you must run `python preprocess_data.py --smiles <file>.smi` to get data ready. You can train with your own SMILES file, or use the provided `zinc250k.smi`, which has SMILES for the ZINC250k dataset. Then call `python train_vae.py`, which will save the trained model as `vae.pt`.