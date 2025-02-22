import numpy as np
def xyz(name, dir):
    with open(f"{dir}/{name}.xyz", "r", encoding="UTF-8") as f:
        data = f.readlines()
    atom_num = int(data[0])

    xyz_data = data[2 : 2 + atom_num]
    xyz_data = [line.strip().split() for line in xyz_data]    
    return atom_num, xyz_data

def line_check(xyz_data):
    tol = 1e-6
    n_atoms = len(xyz_data)
    # If atomic number is less than 2, the molecule is definitely a linear molecule.
    if n_atoms <= 2:
        return True

    coords = np.array([[float(atom[i]) for i in range(0, 3)] for atom in xyz_data])

    # すべてのペア (i, j) を試して直線形かを確認
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            ref_vec = coords[j] - coords[i]
            if np.linalg.norm(ref_vec) < tol:
                continue

            is_linear = True
            for k in range(n_atoms):
                if k == i or k == j:
                    continue
                vec_k = coords[k] - coords[i]
                if np.linalg.norm(vec_k) < tol:
                    continue
                cross_prod = np.cross(ref_vec, vec_k)
                if not np.allclose(cross_prod, 0, atol=tol):
                    is_linear = False
                    break

            if is_linear:
                return True

    return False
