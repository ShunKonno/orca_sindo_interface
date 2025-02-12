def energy(name, dir):
    with open(f"{dir}/{name}.out","r",encoding="UTF-8") as f:
        data = f.readlines()
    
    energy_index = next(i for i, line in enumerate(data) if "Total Energy       :" in line)
    energy = float(data[energy_index].split(":")[1].split("Eh")[0])
    
    return energy

def charge(name, dir):
    with open(f"{dir}/{name}.out","r",encoding="UTF-8") as f:
        data = f.readlines()
    
    charge_index = next(i for i, line in enumerate(data) if "Total Charge" in line)
    charge = float(data[charge_index].split("....")[1].split("\n")[0])

    return charge

def multiplicity(name, dir):
    with open(f"{dir}/{name}.out","r",encoding="UTF-8") as f:
        data = f.readlines()

    mult_index = next(i for i, line in enumerate(data) if "Multiplicity" in line)
    multiplicity = float(data[mult_index].split("....")[1].split("\n")[0])

    return multiplicity

def atomic_number(name, xyz_data, atom_num, dir):
    with open(f"{dir}/{name}.out","r",encoding="UTF-8") as f:
        data = f.readlines()

    nuc_index = next(i for i, line in enumerate(data) if "CARTESIAN COORDINATES (A.U.)" in line)
    inserted_indices = set()

    for i in range(atom_num):
        nuc_num = data[nuc_index + 3 + i].split()[1:3]
        nuc_num[1] = str(int(float(nuc_num[1])))
        for j, atomic in enumerate(xyz_data):
            if nuc_num[0] == atomic[0] and j not in inserted_indices:
                atomic.insert(1, nuc_num[1])
                inserted_indices.add(j)
                break

    xyz_data_full = [
        ", ".join(parts) for parts in xyz_data
    ] 

    return xyz_data_full

def dipole_moment(name, dir):
    with open(f"{dir}/{name}.out","r",encoding="UTF-8") as f:
        data = f.readlines()
        
    di_mo_index = next(i for i, line in enumerate(data) if "Total Dipole Moment" in line)
    dipole_moment = data[di_mo_index].split(":")[1].strip("\n").split()
    dipole_moment_line = ", ".join(val for val in dipole_moment)

    return dipole_moment_line

def polarizability(name, dir):
    with open(f"{dir}/{name}.out","r",encoding="UTF-8") as f:
        data = f.readlines()

    pol_index = next(i for i, line in enumerate(data) if "The raw cartesian tensor (atomic units):" in line)
    pol_row = data[pol_index+1 : pol_index+4]
    pol_line = list(range(6))
    for i,line in enumerate(pol_row):
        nline = line.split()
        if i == 0:
            pol_line[0] = str(nline[0])
            pol_line[1] = str(nline[1])
            pol_line[3] = str(nline[2])
        elif i == 1:
            pol_line[2] = str(nline[1])
            pol_line[4] = str(nline[2])
        elif i == 2:
            pol_line[5] = str(nline[2])
    polarizability = "\n".join(", ".join(str(x) for x in pol_line[i:i+5]) for i in range(0, len(pol_line), 5))

    return polarizability


