#!/usr/bin/env python3
import numpy as np
import os
import sys
from orca_run import main
from grid_xml import main

sys.path.append(os.path.abspath("../orca_sindo_minfo"))
from my_package.hess import dipole, hessian
from my_package.out import energy, charge, multiplicity, dipole_moment, polarizability
from my_package.engrad import engrad
def main(dir_):
    
    orca_run.main()
    grid_xml.main()

    qff_directory = "./mkqff/" + dir_
    input_name = "job"
    with open(f"{qff_directory}/{input_name}.out", "r", encoding="UTF-8") as f:
        data = f.readlines()

    atom_num_index = next(i for i, line in enumerate(data) if line.startswith("Number of atoms"))
    atom_num = int(data[atom_num_index].split()[-1])
    
    dipole_val = dipole(input_name, atom_num, qff_directory)
    hessian_val = hessian(input_name, atom_num, qff_directory)

    energy_val = energy(input_name, qff_directory)
    charge_val = charge(input_name, qff_directory)
    multiplicity_val = multiplicity(input_name, qff_directory)

    dipole_moment_val = dipole_moment(input_name, qff_directory)
    polarizability_val = polarizability(input_name, qff_directory)

    gradient = engrad(input_name, atom_num, qff_directory)
    
    output_path = "../output/minfo.files"
    os.makedirs(output_path, exist_ok=True)

    with open(f"{output_path}/{dir_}.minfo", "w", encoding="UTF-8") as f:
        f.write("# minfo File version 2:\n")
        f.write("#\n")
        f.write("[ Electronic Data ]\n")
        f.write("Energy\n")
        f.write(str(energy_val))
        f.write("\nCharge\n")
        f.write(str(charge_val))
        f.write("\nMultiplicity\n")
        f.write(str(multiplicity_val))
        f.write("\nGradient\n")
        f.write(str(atom_num*3))
        f.write("\n")
        f.write(gradient)
        f.write("\n")
        f.write("Hessian\n")
        f.write(str(int(atom_num**2*(atom_num**2+1)/2)))
        f.write("\n")
        f.write(hessian_val)
        f.write("\nDipole Moment\n3\n")
        f.write(dipole_moment_val)
        f.write("\nPolarizability\n")
        f.write("6\n")
        f.write(polarizability_val)
        f.write("\n")
        f.write("Dipole Derivative\n")
        f.write(str(atom_num*3*3))
        f.write("\n")
        f.write(dipole_val)

if __name__ == "__main__":
    directories = [
    d for d in os.listdir("mkqff/")
    if d.startswith('mkqff') and os.path.isdir(os.path.join("mkqff/", d))
    ]
    for dir_ in directories:
        main(dir_)
