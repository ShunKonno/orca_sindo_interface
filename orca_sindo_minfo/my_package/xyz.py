import numpy as np
def xyz(dir="",name):
    with open(f"{dir}{name}.xyz", "r", encoding="UTF-8") as f:
        data = f.readlines()
    atom_num = int(data[0])

    xyz_data = data[2 : 2 + atom_num]
    xyz_data = [line.strip().split() for line in xyz_data]    
    return atom_num, xyz_data

def line_check(xyz_data):
    tol = 1e-6
    n_atoms = len(xyz_data)
    #If atomic number is less than 2, the molecule is definitely a linear molecule.
    if n_atoms <= 2:
        return True

    coords = np.array([[float(atom[i]) for i in range(1, 4)] for atom in xyz_data])

    ref_vec = None
    for i in range(1, n_atoms):
        vec = coords[i] - coords[0]
        if np.linalg.norm(vec) > tol:
            ref_vec = vec
            break

    if ref_vec is None:
        return True

    for i in range(1, n_atoms):
        vec = coords[i] - coords[0]
        if np.linalg.norm(vec) < tol:
            continue
        cross_prod = np.cross(ref_vec, vec)
        if np.linalg.norm(cross_prod) > tol:
            return False
    return True
