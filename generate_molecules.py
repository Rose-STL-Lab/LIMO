from utils import *
from models import *
import torch
import pytorch_lightning as pl
import numpy as np
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    vae = VAE(max_len=dm.dataset.max_len, vocab_len=len(dm.dataset.symbol_to_idx), latent_dim=1024, embedding_dim=64).to(device)
except NameError:
    raise Exception('No dm.pkl found, please run preprocess_data.py first')
vae.load_state_dict(torch.load('vae.pt'))
vae.eval()


def get_optimized_z(weights, num_mols, num_steps=10):
    models = []
    for prop_name in weights:
        models.append(PropertyPredictor(dm.dataset.max_len * len(dm.dataset.symbol_to_idx)))
        models[-1].load_state_dict(torch.load(f'property_models/{prop_name}.pt'))
        models[-1] = models[-1].to(device)
    z = torch.randn((num_mols, 1024), device=device, requires_grad=True)
    optimizer = optim.Adam([z], lr=0.1)
    losses = []
    for epoch in tqdm(range(num_steps), desc='generating molecules'):
        optimizer.zero_grad()
        loss = 0
        probs = torch.exp(vae.decode(z))
        for i, model in enumerate(models):
            out = model(probs)
            loss += torch.sum(out) * list(weights.values())[i]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return z


def get_prop(prop, x):
    return torch.tensor(props[prop](x), device=device).unsqueeze(1).float()


def run_multiobjective_and_filtering(num_mols, max_sa, min_qed):
    weights = {'binding_affinity': 5, 'sa': 2, 'qed': -8}
    z = get_optimized_z(weights, num_mols)
    with torch.no_grad():
        x = torch.exp(vae.decode(z))
    cycles = get_prop('cycles', x)
    x = x[cycles.flatten() == 0]
    sa = get_prop('sa', x)
    x = x[sa.flatten() < max_sa]
    sa = sa[sa.flatten() < max_sa]
    qed = get_prop('qed', x)
    x = x[qed.flatten() > min_qed]
    qed = qed[qed.flatten() > min_qed]
    binding_affinity = get_prop('binding_affinity', x)
    kd = np.exp(binding_affinity.detach().cpu().numpy() / (0.00198720425864083 * 298.15)).flatten()
    sa = sa.detach().cpu().numpy()
    qed = qed.detach().cpu().numpy()
    return [one_hot_to_smiles(hot) for hot in x], kd, sa, qed


parser = argparse.ArgumentParser()
parser.add_argument('--prop', choices=['logp', 'penalized_logp', 'qed', 'sa', 'binding_affinity', 'multi_objective_binding_affinity'], default='multi_objective_binding_affinity')
parser.add_argument('--num_mols', type=int, default=10000)
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--sa_cutoff', type=float, default=5.5)
parser.add_argument('--qed_cutoff', type=float, default=0.4)
parser.add_argument('--optim_steps', type=int, default=10)
parser.add_argument('--autodock_executable', type=str, default='AutoDock-GPU/bin/autodock_gpu_128wi')
parser.add_argument('--protein_file', type=str, default='1err/1err.maps.fld')
args = parser.parse_args()

props = {'logp': one_hots_to_logp, 
         'penalized_logp': one_hots_to_penalized_logp, 
         'qed': one_hots_to_qed, 
         'sa': one_hots_to_sa, 
         'binding_affinity': lambda x: one_hots_to_affinity(x, args.autodock_executable, args.protein_file),
         'cycles': one_hots_to_cycles}

if args.prop == 'multi_objective_binding_affinity':
    smiles, prop, _, _ = run_multiobjective_and_filtering(args.num_mols, args.sa_cutoff, args.qed_cutoff)
else:
    z = get_optimized_z({args.prop: (1 if args.prop in ('sa', 'binding_affinity') else -1)}, args.num_mols, num_steps=args.optim_steps)
    with torch.no_grad():
        x = torch.exp(vae.decode(z))
    smiles = [one_hot_to_smiles(hot) for hot in x]
    prop = get_prop(args.prop, x).detach().cpu().numpy().flatten()
    
if args.prop in ('sa', 'binding_affinity', 'multi_objective_binding_affinity'):
    for i in np.argpartition(prop, args.top_k)[:args.top_k]:
        if args.prop == 'binding_affinity':
            print(delta_g_to_kd(prop[i]), smiles[i])
        else:
            print(prop[i], smiles[i])
else:
    for i in np.argpartition(prop, -args.top_k)[-args.top_k:]:
        print(prop[i], smiles[i])