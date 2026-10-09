from __future__ import annotations
import os
from tqdm import tqdm
from mace.calculators import mace_mp
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

model = "medium"
device = "cuda"
default_dtype = "float64"
calc = mace_mp(
    model=model,
    device=device,
    default_dtype=default_dtype,
)
dir='./outputs/postproc/merged/'
val_outdir='./outputs/postproc/valid_cifs/'
inval_outdir='./outputs/postproc/invalid_cifs/'
ep_list=[]
valid=[]
min_e, max_e=-1000,100
bar=tqdm(total=len(os.listdir(dir)))
for i in os.listdir(dir):
    path=dir+i
    structure=Structure.from_file(path)
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    ep_list.append(energy)
    if energy>min_e and energy<max_e:
        valid.append(energy)
        structure.to(val_outdir+i)
    else:
        structure.to(inval_outdir+i)
    bar.update(1)
