import os
from utils import *
import argparse


if os.path.exists('dm.pkl'):
    os.remove('dm.pkl')

parser = argparse.ArgumentParser()
parser.add_argument('--smiles', type=str, default='zinc250k.smi')
args = parser.parse_args()
print('Preprocessing..')
dm = MolDataModule(1024, args.smiles)
pickle.dump(dm, open('dm.pkl', 'wb'))
print('Done!')