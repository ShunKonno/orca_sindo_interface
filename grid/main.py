#!/usr/bin/env python3
import subprocess
import re
import math
import numpy as np
import itertools
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

def run_orca_job(grid_xyz_lines, calc_id, grid_title, num):
    energy, dipole = run_orca_calculation(grid_xyz_lines, calc_id)
    grid_flat = flatten_geometry(grid_xyz_lines)
    return (grid_title, num, grid_flat, energy, dipole)

# --- 新規：近傍探索による単一モード値取得 ---
def get_nearest_single_mode_value(mode_index, coupling_coords, one_dim_data):
    if coupling_coords is None:
        return None
    best_tuple = None
    best_distance = float('inf')
    for (q, e_diff, coords_flat) in one_dim_data.get(mode_index, []):
        dist = sum((a - b) ** 2 for a, b in zip(coupling_coords, coords_flat))
        if dist < best_distance:
            best_distance = dist
            best_tuple = (q, e_diff)
    return best_tuple

def get_nearest_single_mode_dipole(mode_index, coupling_coords, one_dim_dipole_data):
    if coupling_coords is None:
        return None
    best_tuple = None
    best_distance = float('inf')
    for (q, dip_diff, coords_flat) in one_dim_dipole_data.get(mode_index, []):
        dist = sum((a - b) ** 2 for a, b in zip(coupling_coords, coords_flat))
        if dist < best_distance:
            best_distance = dist
            best_tuple = (q, dip_diff)
    return best_tuple

# ---------------------------
# Main
# ---------------------------
def main():
    with open(xyz_filename, 'r') as f:
        lines = f.readlines()
    i = 0
    nlines = len(lines)
    calc_counter = 0
    eq_energy = None
    eq_dipole = None
    eq_geometry = None
    pot_data_by_title = {}
    dipole_data_by_title = {}

    # --- Equilibrium (参照状態) の計算 ---
    natoms = int(lines[i].strip())
    eq_title_line = lines[i+1].strip()
    eq_xyz_lines = lines[i+2 : i+2+natoms]
    eq_xyz_np = xyz_block_to_np(eq_xyz_lines)
    eq_xyz_for_linecheck = np_to_linecheck_format(eq_xyz_np)
    is_line = line_check(eq_xyz_for_linecheck)
    eq_geometry = eq_xyz_np.flatten().tolist()
    eq_energy, eq_dipole = run_orca_calculation(eq_xyz_lines, "eq")
    print(f"Equilibrium energy: {eq_energy}")
    print(f"Equilibrium dipole: {eq_dipole}")
    i += 2 + natoms
    calc_counter += 1

    # --- 各グリッド点の計算 ---
    grid_tasks = []
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
        grid_xyz_lines = lines[i+2 : i+2+natoms]
        grid_tasks.append({
            "grid_title": grid_title,
            "num": num,
            "grid_xyz_lines": grid_xyz_lines,
            "calc_id": str(calc_counter),
            "grid_flat": flatten_geometry(grid_xyz_lines)
        })
        calc_counter += 1
        i += 2 + natoms

    with ThreadPoolExecutor() as executor:
        future_to_task = {
            executor.submit(run_orca_job,
                            task["grid_xyz_lines"],
                            task["calc_id"],
                            task["grid_title"],
                            task["num"]): task for task in grid_tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                grid_title, num, grid_flat, energy, dipole = future.result()
            except Exception as exc:
                print(f"calc_{task['calc_id']} ({task['grid_title']}) generated an exception: {exc}")
                continue
            # 各グリッド点でのエネルギー差は、orcaで計算した値から平衡状態のエネルギーを引いたもの
            energy_diff = energy - eq_energy
            dipole_diff = (dipole[0] - eq_dipole[0],
                           dipole[1] - eq_dipole[1],
                           dipole[2] - eq_dipole[2])
            if grid_title not in pot_data_by_title:
                pot_data_by_title[grid_title] = {"num": num, "points": []}
            if grid_title not in dipole_data_by_title:
                dipole_data_by_title[grid_title] = {"num": num, "points": []}
            pot_data_by_title[grid_title]["points"].append((grid_flat, energy_diff))
            dipole_data_by_title[grid_title]["points"].append((grid_flat, dipole_diff))
            print(f"calc_{task['calc_id']} ({grid_title}): ΔE = {energy_diff}, ΔDipole = {dipole_diff}")

    # --- Parse normal modes and masses from Hess file ---
    normal_modes = parse_normal_modes(hess_filename)
    offset = 5 if is_line else 6
    _, masses = parse_atom_masses_from_hess(hess_filename)

    # --- 単一モードの結果（q1, q2, …）をまとめる ---
    one_dim_data = {}
    one_dim_dipole_data = {}
    for grid_title, data in pot_data_by_title.items():
        if re.fullmatch(r"q\d+", grid_title):
            mode_idx = int(grid_title[1:]) - 1
            one_dim_data[mode_idx] = []
            for coords_flat, e_diff in data["points"]:
                disp = mass_weighted_displacement(coords_flat, eq_geometry, masses)
                q_val = 50.0 * dot_product(disp, normal_modes[offset + mode_idx])
                one_dim_data[mode_idx].append((q_val, e_diff, coords_flat))
            one_dim_data[mode_idx].sort(key=lambda x: x[0])
        if grid_title in dipole_data_by_title and re.fullmatch(r"q\d+", grid_title):
            mode_idx = int(grid_title[1:]) - 1
            one_dim_dipole_data[mode_idx] = []
            for coords_flat, dip_diff in dipole_data_by_title[grid_title]["points"]:
                disp = mass_weighted_displacement(coords_flat, eq_geometry, masses)
                q_val = 50.0 * dot_product(disp, normal_modes[offset + mode_idx])
                one_dim_dipole_data[mode_idx].append((q_val, dip_diff, coords_flat))
            one_dim_dipole_data[mode_idx].sort(key=lambda x: x[0])

    # --- Create pot files (エネルギー出力) ---
    for grid_title, data in pot_data_by_title.items():
        num = data["num"]
        points = data["points"]
        mode_names = re.findall(r"q\d+", grid_title)
        sorted_mode_names = sorted(mode_names, key=lambda s: int(s[1:]))
        header_line = "#" + "     ".join([f"{name:16s}" for name in sorted_mode_names] + [f"{'Energy':16s}"])
        mode_indices = extract_mode_indices(grid_title)
        sorted_mode_indices = sorted(mode_indices)
        out_filename = os.path.join(output_dir, f"{grid_title}.pot")
        with open(out_filename, "w") as fout:
            fout.write(f"{basis_info} ({num})\n")
            fout.write("# Number of grids and data\n")
            grid_dims = "     ".join([f"{num:8d}" for _ in range(len(sorted_mode_indices))]) + "     " + f"{1:8d}"
            fout.write(grid_dims + "\n")
            fout.write(header_line + "\n")
            onemode_flag = False
            if re.fullmatch(r"q\d+", grid_title):
                # 単一モードの場合は、すでに計算済みのq値をそのまま利用
                mode_idx = int(grid_title[1:]) - 1
                new_points = [ ((q,), e_diff) for (q, e_diff, _) in one_dim_data[mode_idx] ]
                mid = len(new_points) // 2
                new_points.insert(mid, ((0.0,), 0.0))
                if (0.0, 0.0, eq_geometry) not in one_dim_data[mode_idx]:
                    one_dim_data[mode_idx].append((0.0, 0.0, eq_geometry))
                onemode_flag = True
            else:
                # coupling状態の場合：
                new_points = [(None, pt[1], pt[0]) for pt in points]
                # どれか1つの `q` 値が 0.0 になっている組み合わせを extra_points に追加
                n_modes = len(sorted_mode_indices)
                extra_points = []
                # 各モードの q 値リストを取得
                mode_q_values = [
                    [t[0] for t in one_dim_data.get(mode, [])] for mode in sorted_mode_indices
                ]
                # `one_dim_data` の全組み合わせを生成（カップリングの可能性を考慮）
                for prod in itertools.product(*mode_q_values):
                    q_vector = list(prod)
                    if any(q == 0.0 for q in q_vector):
                        extra_points.append((q_vector, 0.0, None))
                # 重複除去
                unique_extra = {}
                for pt in extra_points:
                    key = tuple(round(q, 8) for q in pt[0])
                    if key not in unique_extra:
                        unique_extra[key] = pt
                extra_points = list(unique_extra.values())
                new_points.extend(extra_points)
            # ④ coupling状態の場合、各グリッド点のxyz座標から対応する単一モードの値を近傍探索で取得し、
            computed_points = []
            for item in new_points:
                if onemode_flag:
                    # 単一モードの場合
                    q_vals, e_diff = item
                    adjusted_e = e_diff
                else:
                    q_vals, e_diff, coords_flat = item
                    if coords_flat is not None:
                        sum_single = 0.0
                        new_q_vals = []
                        for mi in sorted_mode_indices:
                            res = get_nearest_single_mode_value(mi, coords_flat, one_dim_data)
                            if res is not None:
                                q_val, single_e_diff = res
                            else:
                                q_val, single_e_diff = 0.0, 0.0
                            new_q_vals.append(q_val)
                            sum_single += (single_e_diff + eq_energy)
                        q_vals = new_q_vals
                        raw_energy = e_diff + eq_energy
                        adjusted_e = raw_energy - sum_single
                    else:
                        adjusted_e = 0.0
                # もし一つでもq値が0なら、補正後のエネルギーは0にする
                if any(q == 0 for q in q_vals):
                    adjusted_e = 0.0
                computed_points.append((q_vals, adjusted_e))
            # 重複するq値ベクトルの行は1件にまとめる
            unique_points = {}
            for q_vals, energy_val in computed_points:
                key = tuple(round(q, 8) for q in q_vals)
                if key not in unique_points:
                    unique_points[key] = energy_val
            computed_points = [(list(key), val) for key, val in unique_points.items()]
            computed_points.sort(key=lambda tup: tup[0][::-1])
            space_count = 5
            space_pattern = " " * space_count
            for q_values, energy_val in computed_points:
                line_str = re.sub(r"\s+", space_pattern, " ".join([f"{q:16.8f}" for q in q_values] + [f"{energy_val:16.10e}"]))
                fout.write(line_str + "\n")
        print(f"Created: {out_filename}  (Total data points: {len(computed_points)})")

    # --- Create dipole files (双極子モーメント出力) ---
    for grid_title, data in dipole_data_by_title.items():
        num = data["num"]
        points = data["points"]
        mode_names = re.findall(r"q\d+", grid_title)
        sorted_mode_names = sorted(mode_names, key=lambda s: int(s[1:]))
        header_line = "#" + "     ".join([f"{name:16s}" for name in sorted_mode_names] +
                                         [f"{'X':16s}", f"{'Y':16s}", f"{'Z':16s}"])
        mode_indices = extract_mode_indices(grid_title)
        sorted_mode_indices = sorted(mode_indices)
        out_filename = os.path.join(output_dir, f"{grid_title}.dipole")
        with open(out_filename, "w") as fout:
            fout.write(f"{basis_info}\n")
            fout.write("# Number of grids and data\n")
            grid_dims = "     ".join([f"{num:8d}" for _ in range(len(sorted_mode_indices))]) + "     " + f"{3:8d}"
            fout.write(grid_dims + "\n")
            fout.write(header_line + "\n")
            onemode_flag = False
            if re.fullmatch(r"q\d+", grid_title):
                mode_idx = int(grid_title[1:]) - 1
                new_points = [ ((q,), dip_diff) for (q, dip_diff, _) in one_dim_dipole_data[mode_idx] ]
                mid = len(new_points) // 2
                new_points.insert(mid, ((0,), (0.0, 0.0, 0.0)))
                if (0.0, (0.0, 0.0, 0.0), eq_geometry) not in one_dim_dipole_data[mode_idx]:
                    one_dim_dipole_data[mode_idx].append((0.0, (0.0, 0.0, 0.0), eq_geometry))
                onemode_flag = True
            else:
                new_points = [(None, pt[1], pt[0]) for pt in data["points"]]
                n_modes = len(sorted_mode_indices)
                extra_points = []
                mode_q_values = [
                    [t[0] for t in one_dim_dipole_data.get(mode, [])] for mode in sorted_mode_indices
                ]
                # `one_dim_data` の全組み合わせを生成（カップリングの可能性を考慮）
                for prod in itertools.product(*mode_q_values):
                    q_vector = list(prod)
                    if any(q == 0.0 for q in q_vector):
                        extra_points.append((q_vector, 0.0, None))
                # 重複除去
                unique_extra = {}
                for pt in extra_points:
                    key = tuple(round(q, 8) for q in pt[0])
                    if key not in unique_extra:
                        unique_extra[key] = pt
                extra_points = list(unique_extra.values())
                new_points.extend(extra_points)
            computed_points = []
            for item in new_points:
                if onemode_flag:
                    q_vals, dip_diff = item
                    adjusted_dip = dip_diff
                else:
                    q_vals, dip_diff, coords_flat = item
                    if coords_flat is not None:
                        sum_single_dip = np.array([0.0, 0.0, 0.0])
                        new_q_vals = []
                        for mi in sorted_mode_indices:
                            res = get_nearest_single_mode_dipole(mi, coords_flat, one_dim_dipole_data)
                            if res is not None:
                                q_val, single_dip_diff = res
                            else:
                                q_val, single_e_diff = 0.0, 0.0
                            new_q_vals.append(q_val)
                            sum_single_dip += (np.array(single_dip_diff) + np.array(eq_dipole))
                        q_vals = new_q_vals
                        raw_dip = np.array(dip_diff) + np.array(eq_dipole)
                        adjusted_dip = tuple(raw_dip - sum_single_dip)
                    else:
                        adjusted_dip = (0.0, 0.0, 0.0)
                if any(q == 0 for q in q_vals):
                    adjusted_dip = (0.0, 0.0, 0.0)
                computed_points.append((q_vals, adjusted_dip))
            unique_points = {}
            for q_vals, dip_val in computed_points:
                key = tuple(round(q, 8) for q in q_vals)
                if key not in unique_points:
                    unique_points[key] = dip_val
            computed_points = [(list(key), val) for key, val in unique_points.items()]
            computed_points.sort(key=lambda tup: tup[0][::-1])
            for q_values, dip_val in computed_points:
                line_str = "     ".join([f"{q:16.8f}" for q in q_values] +
                                        [f"{dip_val[0]:16.10e}", f"{dip_val[1]:16.10e}", f"{dip_val[2]:16.10e}"])
                fout.write(line_str + "\n")
        print(f"Created: {out_filename}  (Total data points: {len(computed_points)})")

    # --- Create q0.pot and q0.dipole for equilibrium state ---
    q0_filename = os.path.join(output_dir, "q0.pot")
    with open(q0_filename, "w") as fout:
        fout.write("# Number of data\n")
        fout.write("     1 \n")
        fout.write("# Data at the reference geometry\n")
        fout.write(f"  {eq_energy:14.8e}\n")
    print(f"Created: {q0_filename} (Equilibrium energy {eq_energy})")

    q0_dipole_filename = os.path.join(output_dir, "q0.dipole")
    with open(q0_dipole_filename, "w") as fout:
        fout.write("# Number of data\n")
        fout.write("     1 \n")
        fout.write("# Dipole at the reference geometry\n")
        fout.write(f"  {eq_dipole[0]:14.8e}     {eq_dipole[1]:14.8e}     {eq_dipole[2]:14.8e}\n")
    print(f"Created: {q0_dipole_filename} (Equilibrium dipole {eq_dipole})")

if __name__ == "__main__":
    main()
    for file in glob.glob("calc*"):
        os.remove(file)
    print("Complete!")

