from my_package.xyz import xyz
from my_package.hess import atom_weight, dipole, hessian, vibration
from my_package.out import energy, charge, multiplicity, atomic_number, dipole_moment, polarizability
from my_package.engrad import engrad

def extract_info(name, dir=""):
    atom_num, xyz_data = xyz(name, dir)

    xyz_data = atom_weight(name, xyz_data, atom_num, dir)

    dipole_val = dipole(name, atom_num, dir)
    hessian_val = hessian(name, atom_num, dir)

    energy_val = energy(name, dir)
    charge_val = charge(name, dir)
    multiplicity_val = multiplicity(name, dir)
    
    xyz_data_full = atomic_number(name, xyz_data, atom_num, dir)

    dipole_moment_val = dipole_moment(name, dir)
    polarizability_val = polarizability(name, dir)
    
    gradient = engrad(name, atom_num, dir)
    
    trans_freqs,trans_vectors,rota_freqs,rota_vectors,vib_freqs,vib_vectors = vibration(name, xyz_data, atom_num, dir)

    return (atom_num, xyz_data_full, dipole_val, hessian_val, energy_val, charge_val, multiplicity_val, 
            dipole_moment_val, polarizability_val, gradient, vib_freqs, vib_vectors, trans_freqs, trans_vectors,
            rota_freqs, rota_vectors)

def main(name, dir):
    (atom_num, xyz_data_full, dipole, hessian, energy, charge, multiplicity, 
     dipole_moment, polarizability, gradient, vib_freqs, vib_vectors, trans_freqs, trans_vectors,
     rota_freqs, rota_vectors) = extract_info(name, dir)
    os.mkdir("../output/", exist_ok=True)
    with open(f"../output/{name}.minfo", "w", encoding="UTF-8") as f:
        f.write("# minfo File version 2:\n")
        f.write("#\n")
        f.write("[ Atomic Data ]\n")
        f.write(str(atom_num) + "\n")
        for line in xyz_data_full:
            f.write(f" {line}\n")
        f.write("\n")
        f.write("[ Electronic Data ]\n")
        f.write("Energy\n")
        f.write(str(energy))
        f.write("\nCharge\n")
        f.write(str(charge))
        f.write("\nMultiplicity\n")
        f.write(str(multiplicity))
        f.write("\nGradient\n")
        f.write(str(atom_num*3))
        f.write("\n")
        f.write(gradient)
        f.write("\n")
        f.write("Hessian\n")
        f.write(str(int(atom_num*3*(atom_num*3+1)/2)))
        f.write("\n")
        f.write(hessian)
        f.write("\nDipole Moment\n3\n")
        f.write(dipole_moment)
        f.write("\nPolarizability\n")
        f.write("6\n")
        f.write(polarizability)
        f.write("\n")
        f.write("Dipole Derivative\n")
        f.write(str(atom_num*3*3))
        f.write("\n")
        f.write(dipole)
        f.write("\n")
        f.write("\n[ Vibrational Data ]")
        f.write("\nNormal modes")
        f.write("\nTranslational Frequency\n")
        f.write(f"{len(trans_vectors)}\n")
        f.write(trans_freqs)
        f.write("\nTranslational vector\n")
        for i, val in enumerate(trans_vectors):
            f.write(f"T {i}\n")
            f.write(str(atom_num*3))
            f.write("\n")
            v = "\n".join(", ".join(str(x) for x in val[i:i+5]) for i in range(0, len(val), 5))
            f.write(v)
            f.write("\n")
        f.write("Rotational Frequency\n")
        f.write(f"{len(rota_vectors)}\n")
        f.write(rota_freqs)
        f.write("\nRotational vector\n")
        for i, val in enumerate(rota_vectors):
            f.write(f"R {i}\n")
            f.write(str(atom_num*3))
            f.write("\n")
            v = "\n".join(", ".join(str(x) for x in val[i:i+5]) for i in range(0, len(val), 5))
            f.write(v)
            f.write("\n")
        f.write("Vibrational Frequency\n")
        f.write(f"{len(vib_vectors)}")
        f.write("\n")
        f.write(vib_freqs)
        f.write("\nVibrational vector\n")
        for i, val in enumerate(vib_vectors):
            f.write(f"Mode {i}\n")
            f.write(str(atom_num*3))
            f.write("\n")
            v = "\n".join(", ".join(str(x) for x in val[i:i+5]) for i in range(0, len(val), 5))
            f.write(v)
            f.write("\n")        
        f.write("\n") 


if __name__ == "__main__":
    dir = input("Specify the directory where the ORCA output files are located.")
    name = input("Specify the name of the ORCA output file.")
    print(f"Start the process with the file : {dir}/{name}.__")
    main(name, dir)
    print("Complete!")
