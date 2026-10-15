from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
import os, sys
import pandas as pd
from tqdm import tqdm
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from project.utils.dataset_utils import tables_to_crystal
from project.utils.spg_info import (
    element_number,
    spg_wyckoff_degrees_of_freedom,
    wyckoff_label_to_index,
)
from project.utils.utils import extract_wyckoff_orbit_parameters


def _orbit_sort_key(orbit: dict) -> tuple:
    coord = np.mod(np.asarray(orbit["representative_coord"], dtype=np.float64), 1.0)
    coord[np.isclose(coord, 1.0, atol=1e-8)] = 0.0
    return (*np.round(coord, decimals=10).tolist(), int(orbit.get("orbit_id", 0)))


def pack_orbit_features(results: dict, spg_number: int, max_inf_occ: int) -> dict | None:
    dof_map = spg_wyckoff_degrees_of_freedom[str(int(spg_number))]
    labels = sorted(dof_map, key=lambda label: wyckoff_label_to_index.get(label, 10_000))

    discrete_rows = []
    continuous_rows = []
    wyckoff_letters = []
    wyckoff_dofs = []

    for letter in labels:
        dof = int(dof_map[letter])
        orbits = list(results.get(letter, []))
        if len(orbits) > max_inf_occ or (dof == 0 and len(orbits) > 1):
            return None

        for orbit in sorted(orbits, key=_orbit_sort_key):
            species = orbit["species"]
            if species not in element_number:
                return None
            coord = np.mod(np.asarray(orbit["representative_coord"], dtype=np.float32), 1.0)
            coord[np.isclose(coord, 1.0, atol=1e-6)] = 0.0

            discrete_rows.append(int(element_number[species]))
            continuous_rows.append(coord)
            wyckoff_letters.append(str(letter))
            wyckoff_dofs.append(dof)

    orbit_count = len(discrete_rows)
    if orbit_count == 0:
        return None

    discrete_tab = np.asarray(discrete_rows, dtype=np.int64)
    continuous_tab = np.asarray(continuous_rows, dtype=np.float32).reshape(orbit_count, 3)
    wyckoff_dofs_array = np.asarray(wyckoff_dofs, dtype=np.int64)
    if not (len(discrete_tab) == len(continuous_tab) == len(wyckoff_letters) == len(wyckoff_dofs_array)):
        raise RuntimeError("Packed orbit features have inconsistent first dimensions.")

    return {
        "discrete_tab": discrete_tab,
        "continuous_tab": continuous_tab,
        "wyckoff_letters": wyckoff_letters,
        "wyckoff_dofs": wyckoff_dofs_array,
        "orbit_count": orbit_count,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-inf-occ", type=int, default=30)
    args = parser.parse_args()
    raw_dir = os.path.join(PROJECT_DIR, "data", "mp20", "raw")
    save_dir = os.path.join(PROJECT_DIR, "data", "mp20", "processed")
    csv_dir = os.path.join(PROJECT_DIR, "data", "mp20")
    max_orbit={}
    temp={}
    def process(set):   
        file=pd.read_csv(os.path.join(csv_dir, set + '.csv'))
        id, spg_n= file['material_id'].tolist(), file['spacegroup.number'].tolist()
        fail_c2t, fail_t2c, fail_fit, fail_pack=0, 0, 0, 0
        split_raw_dir = os.path.join(raw_dir, set)
        split_save_dir = os.path.join(save_dir, set)
        os.makedirs(split_save_dir, exist_ok=True)
        bar=tqdm(total=len(os.listdir(split_raw_dir)))
        mask=torch.load(os.path.join(PROJECT_DIR, "utils", "mask_temp.pt"), map_location="cpu", weights_only=False)
        for i in os.listdir(split_raw_dir):
            bar.update(1)
            if i!='mp-1206964.cif':
                continue
            structure_path = os.path.join(split_raw_dir, i)
            structure_raw = Structure.from_file(structure_path)
            ret = extract_wyckoff_orbit_parameters(
                structure_raw,
                spg_true=spg_n[id.index(i[:-4])],
            )
            if ret is None:
                fail_c2t+=1
                continue
            structure_full, results, _, info = ret
            packed = pack_orbit_features(results, info['spg_number'], args.max_inf_occ)
            if packed is None:
                fail_pack += 1
                continue
            info['max_orbit'] = max(len(orbits) for orbits in results.values())
            structure = tables_to_crystal(
                packed['discrete_tab'],
                packed['continuous_tab'],
                packed['wyckoff_letters'],
                packed['wyckoff_dofs'],
                info['spg_number'],
                np.array((info['lattice_abc']+info['lattice_angles'])),
                mask=mask,
            )
            if structure is None:
                fail_t2c+=1
                continue
            matcher = StructureMatcher(
                ltol=1e-5,
                stol=1e-5,
                angle_tol=1e-3,
                primitive_cell=False,
                scale=False,
                attempt_supercell=False,
            )

            ret=matcher.fit(structure_full, structure[0])
            if ret is False:
                fail_fit+=1
                continue
            data={'discrete_tab':packed['discrete_tab'],
                'continuous_tab':packed['continuous_tab'],
                'wyckoff_letters':packed['wyckoff_letters'],
                'wyckoff_dofs':packed['wyckoff_dofs'],
                'orbit_count':packed['orbit_count'],
                'spg_number':info['spg_number'],                
                'lattice_system': info['lattice_system'],
                'lattice_matrix': info['lattice_matrix'],
                'lattice_abc': info['lattice_abc'],
                'lattice_angles': info['lattice_angles'],
                'max_orbit': info['max_orbit']}
            max_orbit[i]=info['max_orbit']
            temp[i]={'spg_number':info['spg_number'],                
                     'lattice_system': info['lattice_system'],
                     'lattice_abc': info['lattice_abc'],
                     'lattice_angles': info['lattice_angles'],}
            np.save(os.path.join(split_save_dir, i[:-4]), data)
        print(
            f'fail_c2t:{fail_c2t}, fail_t2c:{fail_t2c}, fail_fit:{fail_fit}, '
            f'fail_pack:{fail_pack}, total:{len(os.listdir(split_raw_dir))}'
        )
    process('train')
    process('test')
    process('val')
    import json
    f=open(os.path.join(csv_dir, "base_info.json"), "w", encoding="utf-8")
    json.dump(temp, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
