
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymatgen.io.vasp import Outcar
from collections import Counter
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import sys,os

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from project.utils.utils import ref_energy

dirs='./outputs/postproc/dftres/'
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
