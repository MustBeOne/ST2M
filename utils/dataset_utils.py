from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure
from collections import defaultdict
import pyxtal
from project.utils.spg_info import spg_wyckoff_degrees_of_freedom, number_element

def analyse_orbit_occupation(discrete_tab, wyckoff_letters):
    posi_orbs=defaultdict(list)
    for n,pos in enumerate(wyckoff_letters):
        pos_atoms = discrete_tab[n]
        posi_orbs[pos].append(number_element[pos_atoms]) 
    return posi_orbs


def project_wyckoffpos(site,template:str,mask):
    if 'x' in template:
        template=template.replace('x',str(site[0]))
    if 'y' in template:
        template=template.replace('y',str(site[1]))
    if 'z' in template:
        template=template.replace('z',str(site[2]))
    coords=template.split(', ')
    proj_coord=np.zeros(3)
    proj_coord[np.nonzero(mask)[0]]=site[np.nonzero(mask)[0]]
    for i in np.nonzero(mask-1)[0]:
        proj_coord[i]=eval(coords[i])
    return proj_coord

def tables_to_crystal(
    discrete_tab: np.ndarray,
    continuous_tab: np.ndarray,
    wyckoff_letters,
    wyckoff_dofs,
    spg_number: int,
    lattice_const:np.ndarray,
    mask,
):
    # Construct a full Structure
    info={}
    info['orbits']=[]
    discrete_tab = np.asarray(discrete_tab, dtype=np.int64).reshape(-1)
    continuous_tab = np.asarray(continuous_tab, dtype=np.float32).reshape(-1, 3)
    wyckoff_letters = [str(letter) for letter in wyckoff_letters]
    wyckoff_dofs = np.asarray(wyckoff_dofs, dtype=np.int64).reshape(-1)
    orbit_count = len(wyckoff_letters)
    if not (len(discrete_tab) == len(continuous_tab) == len(wyckoff_dofs) == orbit_count):
        raise ValueError("Packed discrete, continuous, letter, and dof fields must have equal lengths.")
    group_mask = mask.get(str(spg_number), mask.get(int(spg_number))) if isinstance(mask, dict) else None
    if not isinstance(group_mask, dict):
        raise KeyError(f"No Wyckoff coordinate masks for space group {spg_number}.")
    a, b, c, alpha, beta, gamma = lattice_const
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    structure = Structure.from_spacegroup(spg_number,lattice, [], [])
    G=pyxtal.Group(spg_number)
    posi_orbs = analyse_orbit_occupation(discrete_tab, wyckoff_letters)
    if len(posi_orbs)==0:
        return None
    dof_of_spg=spg_wyckoff_degrees_of_freedom[str(spg_number)]
    fixed_letters = set()
    for letter, saved_dof in zip(wyckoff_letters, wyckoff_dofs.tolist()):
        if letter not in dof_of_spg:
            raise ValueError(f"Wyckoff letter {letter!r} is invalid for space group {spg_number}.")
        dof = int(dof_of_spg[letter])
        if int(saved_dof) != dof:
            raise ValueError(f"Saved dof for {letter} is {saved_dof}, expected {dof}.")
        if dof == 0 and letter in fixed_letters:
            raise ValueError(f"Fixed Wyckoff position {letter} occurs more than once.")
        if dof == 0:
            fixed_letters.add(letter)
    for orb, atoms in posi_orbs.items():
        if len(atoms)>0:
            dof=dof_of_spg[orb]
            if dof != 0:
                wp=G.get_wp_by_letter(orb)
                continuous_rows = continuous_tab[np.where(np.array(wyckoff_letters)==orb)[0].tolist()]
                if orb not in group_mask:
                    raise KeyError(f"No mask for space group {spg_number}, Wyckoff letter {orb}.")
                rep_mask=np.asarray(group_mask[orb], dtype=np.float32).reshape(3)
                if int(np.rint(rep_mask.sum())) != dof:
                    raise ValueError(f"Mask for space group {spg_number}, letter {orb} does not match dof={dof}.")
                n_atom=len(atoms)
                for i in range(n_atom):
                    rep_coord=continuous_rows[i]
                    proj_coord = project_wyckoffpos(rep_coord, wp.__str__().split('\n')[1],rep_mask)
                    pos0 = proj_coord
                    all_coords = wp.apply_ops(pos0)
                    temp={'specie':atoms[i], 'letter':orb, 'rep_coord':pos0.tolist(), 'coord_mask':rep_mask.tolist()}
                    info['orbits'].append(temp)
                    for site in all_coords:
                        structure.append(atoms[i], np.mod(site, 1.0), coords_are_cartesian=False)
            else:
                if len(atoms)>1:
                    print('The number of orbit in non-dof position is not 1! ')
                    return None
                wp=G.get_wp_by_letter(orb)
                templates=wp.__str__().split('\n')[1:]
                sites=[]
                for item in templates:
                    site=eval(item)
                    sites.append(site)
                    structure.append(atoms[0],  np.mod(site, 1.0), coords_are_cartesian=False)
                temp={'specie':atoms[0], 'letter':orb, 'rep_coord':sites[0], 'coord_mask':[0.,0.,0.]}
                info['orbits'].append(temp)
    # structure.merge_sites(mode="delete", tol=0.1)    
    info['lattice']=lattice_const.tolist()
    return structure, info

