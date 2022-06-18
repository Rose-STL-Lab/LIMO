from utils import *
from models import *
import matplotlib.pyplot as plt
import torch
import numpy as np
from rdkit.Chem import MolFromSmiles
import argparse
import time


replacements = ('N', 'O', 'Cl', 'F')

parser = argparse.ArgumentParser()
parser.add_argument('smiles', type=str)
parser.add_argument('--autodock_executable', type=str, default='AutoDock-GPU/bin/autodock_gpu_128wi')
parser.add_argument('--protein_file', type=str, default='1err/1err.maps.fld')
args = parser.parse_args()
base = args.smiles

while True:
    tests = [base]
    for i in range(len(base)):
        if base[i] == 'C':
            for replace in replacements:
                test = list(base)
                test[i] = replace
                test = ''.join(test)
                m = MolFromSmiles(test)
                if m:
                    for other in replacements:
                        if m.HasSubstructMatch(MolFromSmiles(f'{replace}{other}')):
                            break
                        if MolFromSmiles(f'{replace}={other}') and m.HasSubstructMatch(MolFromSmiles(f'{replace}={other}')):
                            break
                    else:
                        tests.append(test)
    tests *= 10
    affins = smiles_to_affinity(tests, args.autodock_executable, args.protein_file)
    base = tests[np.argmin(affins)]
    print(delta_g_to_kd(min(affins)), base)
    time.sleep(2)