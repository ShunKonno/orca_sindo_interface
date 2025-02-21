#!/usr/bin/env python3
import subprocess
import re
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import glob
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(script_dir, "../orca_sindo_minfo"))
sys.path.append(target_dir)
from my_package.xyz import line_check

# ---------------------------
# Settings
# ---------------------------
orca_header_eq = """! B3LYP def2-SVP TightSCF Freq

* xyz 0 1
"""
orca_header_sp = """! B3LYP def2-SVP TightSCF

* xyz 0 1
"""
basis_info = "B3LYP/def2-SVP"
hess_filename = "calc_eq.hess"
xyz_filename = "../output/makeGrid.xyz"
output_dir = os.path.join("..", "output", "pot")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Helper functions
# ---------------------------
def extract_energy(output_file):
    energy = None
    with open(output_file, 'r') as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                m = re.search(r"FINAL SINGLE POINT ENERGY\s+([-+]?\d*\.\d+|\d+)", line)
                if m:
                    energy = float(m.group(1))
                    break
    return energy

def extract_dipole(output_file):
    dipole = None
    with open(output_file, 'r') as f:
        for line in f:
            if "Total Dipole Moment" in line:
                m = re.search(r"Total Dipole Moment\s*:\s*([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)", line)
                if m:
                    dipole = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                    break
    return dipole

def run_orca_calculation(coordinates, calc_id):
    header = orca_header_eq if calc_id == "eq" else orca_header_sp
    input_filename = f"calc_{calc_id}.inp"
    output_filename = f"calc_{calc_id}.out"
    input_content = header + "".join(coordinates) + "*\n"
    with open(input_filename, 'w') as f:
        f.write(input_content)
    orca_path = "orca"
    cmd = f"{orca_path} {input_filename} > {output_filename}"
    subprocess.run(cmd, shell=True, check=True)
    energy = extract_energy(output_filename)
    dipole = extract_dipole(output_filename)
    if calc_id != "eq":
        for fn in (input_filename, output_filename, f"calc_{calc_id}.hess"):
            if os.path.exists(fn):
                os.remove(fn)
    return energy, dipole

def parse_xyz_line(line):
    tokens = line.split()
    atom = tokens[0]
    coords = list(map(float, tokens[1:4]))
    return atom, coords

def xyz_block_to_np(xyz_lines):
    coords_list = []
    for line in xyz_lines:
        _, coord = parse_xyz_line(line)
        coords_list.append(coord)
    return np.array(coords_list)

def np_to_linecheck_format(xyz_np):
    out = []
    for row in xyz_np:
        x, y, z = row
        out.append(["X", str(x), str(y), str(z)])
    return out

def flatten_geometry(xyz_lines):
    flat = []
    for line in xyz_lines:
        _, coords = parse_xyz_line(line)
        flat.extend(coords)
    return flat

def dot_product(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2))

def parse_normal_modes(hess_filename):
    with open(hess_filename, 'r') as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("$normal_modes"):
            start = i
            break
    dims_line = lines[start+1].strip()
    dims = dims_line.split()
    nrows = int(dims[0])
    ncols = int(dims[1])
    matrix = [[None]*ncols for _ in range(nrows)]
    line_index = start + 2
    cols_processed = 0
    while cols_processed < ncols:
        while line_index < len(lines) and not lines[line_index].strip():
            line_index += 1
        header_line = lines[line_index].strip()
        block_cols = [int(tok) for tok in header_line.split()]
        block_ncols = len(block_cols)
        line_index += 1
        for row in range(nrows):
            row_line = lines[line_index].strip()
            line_index += 1
            tokens = row_line.split()
            values = list(map(float, tokens[1:]))
            for j, col in enumerate(block_cols):
                matrix[row][col] = values[j]
        cols_processed += block_ncols
    modes = []
    for col in range(ncols):
        mode_col = [matrix[row][col] for row in range(nrows)]
        modes.append(mode_col)
    return modes

def extract_mode_indices(grid_title):
    qs = re.findall(r"q(\d+)", grid_title)
    return [int(q) - 1 for q in qs]

def parse_atom_masses_from_hess(hess_filename):
    with open(hess_filename, 'r') as f:
        lines = f.readlines()
    start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("$atoms"):
            start_index = i
            break
    n_atoms = int(lines[start_index+1].strip())
    masses = []
    for j in range(n_atoms):
        tokens = lines[start_index+2+j].split()
        mass = float(tokens[1])
        masses.append(mass)
    return n_atoms, masses

def mass_weighted_displacement(coords, eq_coords, masses):
    if len(coords) != len(eq_coords):
        raise ValueError("Coordinate sizes do not match.")
    disp = []
    for i in range(0, len(coords), 3):
        dx = coords[i]   - eq_coords[i]
        dy = coords[i+1] - eq_coords[i+1]
        dz = coords[i+2] - eq_coords[i+2]
        m = masses[i // 3]
        disp.extend([dx * math.sqrt(m), dy * math.sqrt(m), dz * math.sqrt(m)])
    return disp

# --- グリッドの補完用関数 ---
def generate_multi_indices(G, N):
    def rec(level, current):
        if level == 0:
            yield tuple(current)
        else:
            for i in range(G):
                current.append(i)
                yield from rec(level - 1, current)
                current.pop()
    all_indices = list(rec(N, []))
    all_indices.sort(key=lambda tup: tup[::-1])
    return all_indices

def fill_missing_tasks(tasks, G, q_count, eq_geometry):
    total_count = G ** q_count
    tasks_dict = {}
    for t in tasks:
        if t.get("grid_index") is not None:
            tasks_dict[t["grid_index"]] = t
    complete_tasks = []
    for i in range(total_count):
        if i in tasks_dict:
            complete_tasks.append(tasks_dict[i])
        else:
            lower = None
            higher = None
            lower_index = None
            higher_index = None
            # 下側の既知の値を探す
            for j in range(i-1, -1, -1):
                if j in tasks_dict:
                    lower = tasks_dict[j]
                    lower_index = j
                    break
            # 上側の既知の値を探す
            for j in range(i+1, total_count):
                if j in tasks_dict:
                    higher = tasks_dict[j]
                    higher_index = j
                    break
            if lower is not None and higher is not None:
                # 線形補間の割合 t = (i - lower_index) / (higher_index - lower_index)
                t = (i - lower_index) / (higher_index - lower_index)
                interp_coords = [a*(1-t) + b*t for a, b in zip(lower["grid_flat"], higher["grid_flat"])]
            elif lower is not None:
                interp_coords = lower["grid_flat"]
            elif higher is not None:
                interp_coords = higher["grid_flat"]
            else:
                interp_coords = eq_geometry
            new_task = {
                "grid_title": tasks[0]["grid_title"] if tasks else "",
                "num": G,
                "grid_index": i,
                "grid_flat": interp_coords,
                "calc_id": "interp",
                "energy_diff": 0.0,
                "dipole_diff": (0.0, 0.0, 0.0)
            }
            complete_tasks.append(new_task)
    return complete_tasks


# --- Wrapper for parallel ORCA jobs ---
def run_orca_job(grid_xyz_lines, calc_id, grid_title, num, grid_index):
    energy, dipole = run_orca_calculation(grid_xyz_lines, calc_id)
    grid_flat = flatten_geometry(grid_xyz_lines)
    return {
        "grid_title": grid_title,
        "num": num,
        "grid_index": grid_index,
        "grid_flat": grid_flat,
        "calc_id": calc_id,
        "energy_diff": energy - eq_energy,
        "dipole_diff": (dipole[0] - eq_dipole[0],
                        dipole[1] - eq_dipole[1],
                        dipole[2] - eq_dipole[2])
    }

# ---------------------------
# Main
# ---------------------------
def main():
    global eq_energy, eq_dipole
    with open(xyz_filename, 'r') as f:
        lines = f.readlines()
    i = 0
    nlines = len(lines)
    calc_counter = 0
    eq_energy = None
    eq_dipole = None
    eq_geometry = None
    tasks_by_title = {}

    # --- Equilibrium geometry の読み込み ---
    natoms = int(lines[i].strip())
    eq_title_line = lines[i+1].strip()
    eq_xyz_lines = lines[i+2:i+2+natoms]
    eq_xyz_np = xyz_block_to_np(eq_xyz_lines)
    eq_xyz_for_linecheck = np_to_linecheck_format(eq_xyz_np)
    is_line = line_check(eq_xyz_for_linecheck)
    eq_geometry = eq_xyz_np.flatten().tolist()
    eq_energy, eq_dipole = run_orca_calculation(eq_xyz_lines, "eq")
    print(f"Equilibrium energy: {eq_energy}")
    print(f"Equilibrium dipole: {eq_dipole}")
    i += 2 + natoms
    calc_counter += 1

    # --- mkg- ブロックの読み込み ---
    while i < nlines:
        if not lines[i].strip().isdigit():
            i += 1
            continue
        natoms = int(lines[i].strip())
        title_line = lines[i+1].strip()
        remainder = title_line[len("mkg-"):]
        parts = remainder.split("-")
        grid_title = parts[0]
        num = int(parts[1])
        grid_index = int(parts[2]) if len(parts) >= 3 else None
        grid_xyz_lines = lines[i+2:i+2+natoms]
        task = {
            "grid_title": grid_title,
            "num": num,
            "grid_index": grid_index,
            "grid_xyz_lines": grid_xyz_lines,
            "calc_id": str(calc_counter),
            "grid_flat": flatten_geometry(grid_xyz_lines)
        }
        tasks_by_title.setdefault(grid_title, []).append(task)
        calc_counter += 1
        i += 2 + natoms

    # --- 並列 ORCA 計算の実行 ---
    results_by_title = {}
    with ThreadPoolExecutor() as executor:
        futures = []
        for grid_title, tasks in tasks_by_title.items():
            for task in tasks:
                future = executor.submit(run_orca_job,
                                         task["grid_xyz_lines"],
                                         task["calc_id"],
                                         task["grid_title"],
                                         task["num"],
                                         task.get("grid_index"))
                futures.append((grid_title, future))
        for grid_title, future in futures:
            try:
                result = future.result()
            except Exception as exc:
                print(f"Error in grid {grid_title}: {exc}")
                continue
            results_by_title.setdefault(grid_title, []).append(result)
            print(f"calc_{result['calc_id']} ({grid_title}): ΔE = {result['energy_diff']}, ΔDipole = {result['dipole_diff']}")

    # --- 各グループについて、全スロット (G^(q_count)) 分に補完 ---
    complete_tasks_by_title = {}
    for grid_title, tasks in results_by_title.items():
        q_count = len(re.findall(r"q\d+", grid_title))
        G = tasks[0]["num"]
        complete = fill_missing_tasks(tasks, G, q_count, eq_geometry)
        complete_tasks_by_title[grid_title] = complete

    # --- ファイル出力 ---
    # 一度、normal_modes を一括で読み出す
    nm = parse_normal_modes(hess_filename)
    # offset: 非線形なら6、線形なら5
    offset = 6 if not is_line else 5
    _, masses = parse_atom_masses_from_hess(hess_filename)

    # pot ファイル出力（各行は、q座標2個（例：q3q1ならq1, q3）＋エネルギーの3列）
    for grid_title, tasks in complete_tasks_by_title.items():
        num = tasks[0]["num"]
        mode_names = sorted(re.findall(r"q\d+", grid_title), key=lambda s: int(s[1:]))
        # sorted_mode_indices は、抽出した q番号（0ベース）を昇順に
        sorted_mode_indices = sorted(extract_mode_indices(grid_title))
        header_line = "#" + "     ".join([f"{name:16s}" for name in mode_names] + [f"{'Energy':16s}"])
        out_filename = os.path.join(output_dir, f"{grid_title}.pot")
        with open(out_filename, "w") as fout:
            fout.write(f"{basis_info} ({num})\n")
            fout.write("# Number of grids and data\n")
            grid_dims = "     ".join([f"{num:8d}" for _ in range(len(mode_names))]) + "     " + f"{1:8d}"
            fout.write(grid_dims + "\n")
            fout.write(header_line + "\n")
            for task in tasks:
                disp = mass_weighted_displacement(task["grid_flat"], eq_geometry, masses)
                q_values = []
                for idx in sorted_mode_indices:
                    q_val = 50.0 * dot_product(disp, nm[offset + idx])
                    q_values.append(q_val)
                line_str = "     ".join([f"{q:16.8f}" for q in q_values] +
                                         [f"{task['energy_diff']:16.10e}"])
                fout.write(line_str + "\n")
        print(f"Created: {out_filename}  (Data points: {len(tasks)})")

    # dipole ファイル出力（同様に、q座標2個＋X, Y, Z の合計5列）
    for grid_title, tasks in complete_tasks_by_title.items():
        num = tasks[0]["num"]
        mode_names = sorted(re.findall(r"q\d+", grid_title), key=lambda s: int(s[1:]))
        sorted_mode_indices = sorted(extract_mode_indices(grid_title))
        header_line = "#" + "     ".join([f"{name:16s}" for name in mode_names] +
                                         [f"{'X':16s}", f"{'Y':16s}", f"{'Z':16s}"])
        out_filename = os.path.join(output_dir, f"{grid_title}.dipole")
        with open(out_filename, "w") as fout:
            fout.write(f"{basis_info}\n")
            fout.write("# Number of grids and data\n")
            grid_dims = "     ".join([f"{num:8d}" for _ in range(len(mode_names))]) + "     " + f"{3:8d}"
            fout.write(grid_dims + "\n")
            fout.write(header_line + "\n")
            for task in tasks:
                disp = mass_weighted_displacement(task["grid_flat"], eq_geometry, masses)
                q_values = []
                for idx in sorted_mode_indices:
                    q_val = 50.0 * dot_product(disp, nm[offset + idx])
                    q_values.append(q_val)
                line_str = "     ".join([f"{q:16.8f}" for q in q_values] +
                                        [f"{task['dipole_diff'][0]:16.10e}",
                                         f"{task['dipole_diff'][1]:16.10e}",
                                         f"{task['dipole_diff'][2]:16.10e}"])
                fout.write(line_str + "\n")
        print(f"Created: {out_filename}  (Data points: {len(tasks)})")

    # q0 ファイル出力
    q0_filename = os.path.join(output_dir, "q0.pot")
    with open(q0_filename, "w") as fout:
        fout.write("# Number of data\n     1 \n")
        fout.write("# Data at the reference geometry\n")
        fout.write(f"  {eq_energy:14.8e}\n")
    print(f"Created: {q0_filename} (Equilibrium energy {eq_energy})")
    q0_dipole_filename = os.path.join(output_dir, "q0.dipole")
    with open(q0_dipole_filename, "w") as fout:
        fout.write("# Number of data\n     1 \n")
        fout.write("# Dipole at the reference geometry\n")
        fout.write(f"  {eq_dipole[0]:14.8e}     {eq_dipole[1]:14.8e}     {eq_dipole[2]:14.8e}\n")
    print(f"Created: {q0_dipole_filename} (Equilibrium dipole {eq_dipole})")

if __name__ == "__main__":
    main()
    for file in glob.glob("calc*"):
        os.remove(file)
    print("Complete!")

