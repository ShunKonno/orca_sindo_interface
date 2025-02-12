import numpy as np
def atom_weight(dir="", name, xyz_data, atom_num):
    with open(f"{dir}{name}.hess", "r", encoding="UTF-8") as f:
        data = f.readlines()

    atom_weight_index = data.index("$atoms\n")+2

    atom_weight_line = data[atom_weight_index : atom_weight_index + atom_num]
    atom_weight_line = [line.strip().split() for line in atom_weight_line]

    for i, (atom, weight_data) in enumerate(zip(xyz_data, atom_weight_line)):
        if atom[0] == weight_data[0]: 
            weight = weight_data[1]  
            atom.insert(1, weight)  
    
    return xyz_data

def dipole(dir="", name, atom_num):
    with open(f"{dir}{name}.hess", "r", encoding="UTF-8") as f:
        data = f.readlines()
    dipole_index = data.index("$dipole_derivatives\n")+2

    dipole_line = data[dipole_index : dipole_index + atom_num*3]
    dipole_line = [line.strip().split() for line in dipole_line]

    dipole_data = [
        dipole_line[row][col]
        for col in range(3)
        for row in range(atom_num*3)
    ]
    dipole = "\n".join(", ".join(dipole_data[i:i+5]) for i in range(0, len(dipole_data), 5))
    return dipole

def hessian(dir="", name, atom_num):
    with open(f"{dir}{name}.hess", "r", encoding="UTF-8") as f:
        data = f.readlines()
    hessian_start = data.index("$hessian\n") + 3
    N = 3 * atom_num

    hessian_end = hessian_start + 1 + ((atom_num**2 // 5 + 1) * atom_num**2)
    hessian_lines = data[hessian_start:hessian_end]
    
    def extract_numbers(data):
        all_numbers = []
        for line in data:
            parts = line.split()
            numbers = [float(p) for p in parts if "E" in p or "." in p]
            if numbers:
                all_numbers.append(numbers)
        return all_numbers

    row_matrix = extract_numbers(hessian_lines)
    
    group_size = atom_num*3
    num_groups = len(row_matrix) // group_size
    groups = [row_matrix[i * group_size:(i + 1) * group_size] for i in range(num_groups)]
    
    matrix = []
    for i in range(group_size):
        combined_row = []
        for group in groups:
            combined_row.extend(group[i])
        matrix.append(combined_row)

    custom_order = []
    for i in range(len(matrix)):
        for j in range(i + 1):
            custom_order.append(str(matrix[j][i]))
    
    return custom_order
