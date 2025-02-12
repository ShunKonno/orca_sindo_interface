import numpy as np
from .xyz import line_check
def atom_weight(name, xyz_data, atom_num, dir):
    with open(f"{dir}/{name}.hess", "r", encoding="UTF-8") as f:
        data = f.readlines()

    atom_weight_index = data.index("$atoms\n")+2

    atom_weight_line = data[atom_weight_index : atom_weight_index + atom_num]
    atom_weight_line = [line.strip().split() for line in atom_weight_line]

    for i, (atom, weight_data) in enumerate(zip(xyz_data, atom_weight_line)):
        if atom[0] == weight_data[0]: 
            weight = weight_data[1]  
            atom.insert(1, weight)  
    
    return xyz_data

def dipole(name, atom_num, dir):
    with open(f"{dir}/{name}.hess", "r", encoding="UTF-8") as f:
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

def hessian(name, atom_num, dir):
    with open(f"{dir}/{name}.hess", "r", encoding="UTF-8") as f:
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
    hessian_data = "\n".join(", ".join(custom_order[i:i+5]) for i in range(0, len(custom_order), 5)) 
    return hessian_data

def vibration(name, xyz_data, atom_num, dir):
    with open(f"{dir}/{name}.hess", "r", encoding="UTF-8") as f:
        data = f.readlines()
    
    rot_num = 2 if line_check(xyz_data) else 3
    
    vib_freq_index = data.index("$vibrational_frequencies\n") + 2

    vib_freq_lines = data[vib_freq_index : vib_freq_index + atom_num * 3]

    vib_freq_all = [line.strip().split()[1] for line in vib_freq_lines]
    vib_freqs = vib_freq_all[3 + rot_num:]

    start_index = None
    for i, line in enumerate(data):
        if line.strip().startswith('$normal_modes'):
            start_idx = i
            break
    dims_line = data[start_idx + 1].strip()
    dims_tokens = dims_line.split()
    
    nrows, ncols = int(dims_tokens[0]), int(dims_tokens[1])
    matrix = np.zeros((nrows, ncols))

    current_col = 0
    line_idx = start_idx + 2
    while line_idx < len(data) and current_col < ncols:
        header_line = data[line_idx].strip()
        if header_line == "":
            line_idx += 1
            continue
        header_tokens = header_line.split()
        num_block_cols = len(header_tokens)

        for i in range(nrows):
            data_line = data[line_idx + 1 + i].strip()
            if data_line == "":
                continue
            tokens = data_line.split()
            for j in range(num_block_cols):
                matrix[i, current_col + j] = float(tokens[j+1])
        current_col += num_block_cols
        line_idx += (1 + nrows)
        
        if line_check(xyz_data):
            num_vib = 3 * atom_num - 5
        else:
            num_vib = 3 * atom_num - 6

        vib_modes = matrix[:, -num_vib:]
        vectors = [np.array(vib_modes)[:, i] for i in range(np.array(vib_modes).shape[1])]

    return vib_freqs, vectors
