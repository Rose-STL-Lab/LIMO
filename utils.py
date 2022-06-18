import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import subprocess
import pickle
import selfies as sf
import csv
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import MolFromSmiles, QED
from sascorer import calculateScore
import time
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

delta_g_to_kd = lambda x: math.exp(x / (0.00198720425864083 * 298.15))


class MolDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, file):
        super(MolDataModule, self).__init__()
        self.batch_size = batch_size
        self.dataset = Dataset(file)
        self.train_data, self.test_data = random_split(self.dataset, [int(round(len(self.dataset) * 0.8)), int(round(len(self.dataset) * 0.2))])
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, drop_last=True, num_workers=16, pin_memory=True)
    
    
class PropDataModule(pl.LightningDataModule):
    def __init__(self, x, y, batch_size):
        super(PropDataModule, self).__init__()
        self.batch_size = batch_size
        self.dataset = TensorDataset(x, y)
        self.train_data, self.test_data = random_split(self.dataset, [int(round(len(self.dataset) * 0.9)), int(round(len(self.dataset) * 0.1))])
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, drop_last=True)
        


class Dataset(Dataset):
    def __init__(self, file):
        selfies = [sf.encoder(line.split()[0]) for line in open(file, 'r')]
        self.alphabet = set()
        for s in selfies:
            self.alphabet.update(sf.split_selfies(s))
        self.alphabet = ['[nop]'] + list(sorted(self.alphabet))
        self.max_len = max(len(list(sf.split_selfies(s))) for s in selfies)
        self.symbol_to_idx = {s: i for i, s in enumerate(self.alphabet)}
        self.idx_to_symbol = {i: s for i, s in enumerate(self.alphabet)}
        self.encodings = [[self.symbol_to_idx[symbol] for symbol in sf.split_selfies(s)] for s in selfies]
        
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, i):
        return torch.tensor(self.encodings[i] + [self.symbol_to_idx['[nop]'] for i in range(self.max_len - len(self.encodings[i]))])
    
    
def smiles_to_indices(smiles):
    encoding = [dm.dataset.symbol_to_idx[symbol] for symbol in sf.split_selfies(sf.encoder(smiles))]
    return torch.tensor(encoding + [dm.dataset.symbol_to_idx['[nop]'] for i in range(dm.dataset.max_len - len(encoding))])


def smiles_to_one_hot(smiles):
    out = torch.zeros((dm.dataset.max_len, len(dm.dataset.symbol_to_idx)))
    for i, index in enumerate(smiles_to_indices(smiles)):
        out[i][index] = 1
    return out.flatten()


def smiles_to_z(smiles, vae):
    zs = torch.zeros((len(smiles), 1024), device=device)
    for i, smile in enumerate(tqdm(smiles)):
        target = smiles_to_one_hot(smile).to(device)
        z = vae.encode(smiles_to_indices(smile).unsqueeze(0).to(device))[0].detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=0.1)
        for epoch in range(10000):
            optimizer.zero_grad()
            loss = torch.mean((torch.exp(vae.decode(z)[0]) - target) ** 2)
            loss.backward()
            optimizer.step()
        zs[i] = z.detach()
    return zs
    
    
def one_hots_to_filter(hots):
    f = open('filter/to_filter.csv', 'w')
    for i, hot in enumerate(hots):
        f.write(f'{one_hot_to_smiles(hot)} {i}\n')
    f.close()
    subprocess.run('rd_filters filter --in filter/to_filter.csv --prefix filter/out', shell=True, stderr=subprocess.DEVNULL)
    out = []
    for row in csv.reader(open('filter/out.csv', 'r')):
        if row[0] != 'SMILES':
            out.append(int(row[2] == 'OK'))
    return out


def one_hots_to_logp(hots):
    logps = []
    for i, hot in enumerate(hots):
        smile = one_hot_to_smiles(hot)
        try:
            logps.append(MolLogP(MolFromSmiles(smile)))
        except:
            logps.append(0)
    return logps


def one_hots_to_qed(hots):
    qeds = []
    for i, hot in enumerate(tqdm(hots, desc='calculating QED')):
        smile = one_hot_to_smiles(hot)
        mol = MolFromSmiles(smile)
        qeds.append(QED.qed(mol))
    return qeds


def one_hots_to_sa(hots):
    sas = []
    for i, hot in enumerate(tqdm(hots, desc='calculating SA')):
        smile = one_hot_to_smiles(hot)
        mol = MolFromSmiles(smile)
        sas.append(calculateScore(mol))
    return sas


def one_hots_to_cycles(hots):
    cycles = []
    for hot in tqdm(hots, desc='counting undesired cycles'):
        smile = one_hot_to_smiles(hot)
        mol = MolFromSmiles(smile)
        cycle_count = 0
        for ring in mol.GetRingInfo().AtomRings():
            if not (4 < len(ring) < 7):
                cycle_count += 1
        cycles.append(cycle_count)
    return cycles


def one_hots_to_penalized_logp(hots):
    logps = []
    for i, hot in enumerate(hots):
        smile = one_hot_to_smiles(hot)
        mol = MolFromSmiles(smile)
        penalized_logp = MolLogP(mol) - calculateScore(mol)
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) > 6:
                penalized_logp -= 1
        logps.append(penalized_logp)
    return logps


def smiles_to_penalized_logp(smiles):
    logps = []
    for i, smile in enumerate(smiles):
        mol = MolFromSmiles(smile)
        penalized_logp = MolLogP(mol) - calculateScore(mol)
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) > 6:
                penalized_logp -= 1
        logps.append(penalized_logp)
    return logps


def one_hots_to_affinity(hots, autodock, protein_file, num_devices=torch.cuda.device_count()):
    return smiles_to_affinity([one_hot_to_smiles(hot) for hot in hots], autodock, protein_file, num_devices=num_devices)


def smiles_to_affinity(smiles, autodock, protein_file, num_devices=torch.cuda.device_count()):
    if not os.path.exists('ligands'):
        os.mkdir('ligands')
    if not os.path.exists('outs'):
        os.mkdir('outs')
    subprocess.run('rm core.*', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.xml', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm outs/*.dlg', shell=True, stderr=subprocess.DEVNULL)
    subprocess.run('rm -rf ligands/*', shell=True, stderr=subprocess.DEVNULL)
    for device in range(num_devices):
        os.mkdir(f'ligands/{device}')
    device = 0
    for i, hot in enumerate(tqdm(smiles, desc='preparing ligands')):
        subprocess.Popen(f'obabel -:"{smiles[i]}" -O ligands/{device}/ligand{i}.pdbqt -p 7.4 --partialcharge gasteiger --gen3d', shell=True, stderr=subprocess.DEVNULL)
        device += 1
        if device == num_devices:
            device = 0
    while True:
        total = 0
        for device in range(num_devices):
            total += len(os.listdir(f'ligands/{device}'))
        if total == len(smiles):
            break
    time.sleep(1)
    print('running autodock..')
    if len(smiles) == 1:
        subprocess.run(f'{autodock} -M {protein_file} -s 0 -L ligands/0/ligand0.pdbqt -N outs/ligand0', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        ps = []
        for device in range(num_devices):
            ps.append(subprocess.Popen(f'{autodock} -M {protein_file} -s 0 -B ligands/{device}/ligand*.pdbqt -N ../../outs/ -D {device + 1}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        stop = False
        while not stop: 
            for p in ps:
                stop = True
                if p.poll() is None:
                    time.sleep(1)
                    stop = False
    affins = [0 for _ in range(len(smiles))]
    for file in tqdm(os.listdir('outs'), desc='extracting binding values'):
        if file.endswith('.dlg') and '0.000   0.000   0.000  0.00  0.00' not in open(f'outs/{file}').read():
            affins[int(file.split('ligand')[1].split('.')[0])] = float(subprocess.check_output(f"grep 'RANKING' outs/{file} | tr -s ' ' | cut -f 5 -d ' ' | head -n 1", shell=True).decode('utf-8').strip())
    return [min(affin, 0) for affin in affins]

                
def one_hot_to_selfies(hot):
    return ''.join([dm.dataset.idx_to_symbol[idx.item()] for idx in hot.view((dm.dataset.max_len, -1)).argmax(1)]).replace(' ', '')


def one_hot_to_smiles(hot):
    return sf.decoder(one_hot_to_selfies(hot))


if os.path.exists('dm.pkl'):
    dm = pickle.load(open('dm.pkl', 'rb'))