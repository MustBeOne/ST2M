
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import torch
import json
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from project.utils import config as cfg
from project.generate import lattice6_recover, wyckoff_sample_to_tables
import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test joint D3PM/DDPM U-NET crystal diffusion.")
    cfg.add_common_args(parser)
    cfg.add_model_args(parser)
    cfg.add_test_args(parser)
    return parser

def save_wyckoff_generated_cifs(samples: dict[str, Any], cif_dir:Path, output_dir:Path, parser) -> None:
    from project.utils.dataset_utils import tables_to_crystal

    cif_dir.mkdir(parents=True, exist_ok=True)
    file_list = os.listdir(cif_dir)
    for file_name in file_list:
        os.remove(cif_dir / file_name)
    total={}
    saved = 0
    num_save=len(samples['space_group'])
    mask = torch.load(parser.spacegroup_template_path, map_location="cpu", weights_only=False)
    for i in range(num_save):
        spg = int(samples["space_group"][i].item())
        discrete_tab, continuous_tab, wyckoff_letters, wyckoff_dofs = wyckoff_sample_to_tables(
            samples, i
        )
        if len(discrete_tab) == 0:
            continue
        lattice_matrix = lattice6_recover(samples["lattice"][i]).numpy()
        try:
            ret = tables_to_crystal(
                discrete_tab=discrete_tab,
                continuous_tab=continuous_tab,
                wyckoff_letters=wyckoff_letters,
                wyckoff_dofs=wyckoff_dofs,
                spg_number=spg,
                lattice_const=lattice_matrix,
                mask=mask,
            )
            if ret is None:
                continue
            structure, info=ret
            structure.to(filename=str(cif_dir / f"sample_{i:04d}_sg{spg}.cif"))
            total[f"sample_{i:04d}_sg{spg}.cif"]=info
            saved += 1
        except Exception as exc:
            print(f"skip CIF sample {i}: {exc}")    
        with open(output_dir / "total.json", "w+", encoding="utf-8") as f:
            json.dump(
            total,
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"saved CIFs: {saved}/{num_save} -> {cif_dir}")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
output_dir=PROJECT_ROOT / 'outputs' / 'samples'
samples_dir=output_dir / 'samples_test.pt'
cif_dir= output_dir / 'cif'

samples=torch.load(samples_dir)['samples']
parser=build_parser().parse_args()
save_wyckoff_generated_cifs(samples, cif_dir, output_dir, parser)
