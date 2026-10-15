from pathlib import Path


from pymatgen.io.vasp import Outcar
from collections import Counter
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import os


from st2m.utils.utils import ref_energy
from st2m.utils import config as cfg

def main() -> None:
    dirs = str(cfg.OUTPUT_DIR / 'postproc' / 'dftres') + os.sep
    for i in os.listdir(dirs):
        if 'OUTCAR' in os.listdir(dirs+i):
            outcar = Outcar(dirs+i+'/OUTCAR')
            energy = outcar.final_energy
            structure=Structure.from_file(dirs+i+'/POSCAR')
            adaptor = AseAtomsAdaptor()
            atoms = adaptor.get_atoms(structure)
            element_counts = Counter(atoms.get_chemical_symbols())
            reference_total_energy = 0.0
            for element, number in element_counts.items():
                if element not in ref_energy or not ref_energy[element]:
                    pas=True
                    continue
                reference_total_energy += (number * ref_energy[element])
            formation_energy = (energy - reference_total_energy)
            formation_energy_per_atom = (formation_energy / len(atoms))
    
            print("DFT formation_energy =", formation_energy, "eV")


if __name__ == "__main__":
    main()
