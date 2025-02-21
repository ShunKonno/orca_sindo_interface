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
# ORCA input header (with Freq) for the equilibrium geometry calculation (first run)
orca_header_eq = """! B3LYP def2-SVP TightSCF Freq

* xyz 0 1
"""

# ORCA input header (no Freq) for single-point energy calculations
orca_header_sp = """! B3LYP def2-SVP TightSCF

* xyz 0 1
"""

# Basis information to be written at the top of the pot files
basis_info = "B3LYP/def2-SVP"
hess_filename = "calc_eq.hess"

# Input xyz file name
xyz_filename = "../output/makeGrid.xyz"

# Output directory
output_dir = os.path.join("..", "output", "pot")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# Helper functions
# ---------------------------
def extract_energy(output_file):
    """Extract FINAL SINGLE POINT ENERGY from ORCA output file"""
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
    """Extract Total Dipole Moment (X, Y, Z) from ORCA output file"""
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
    """
    Create an ORCA input file from coordinates (list of strings) and run the calculation.
    """
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
    """Parse one line in xyz format and return (atom_name, (x,y,z))"""
    tokens = line.split()
    atom = tokens[0]
    coords = list(map(float, tokens[1:4]))
    return atom, coords

def xyz_block_to_np(xyz_lines):
    """
    Convert a block of xyz-format lines into a NumPy array.
    """
    coords_list = []
    for line in xyz_lines:
        _, coord = parse_xyz_line(line)
        coords_list.append(coord)
    return np.array(coords_list)  # shape = (N, 3)

def np_to_linecheck_format(xyz_np):
    """
    Convert a NumPy array of shape (N,3) into the format required by line_check.
    Each element is [symbol, x, y, z]. Here, symbol is fixed to "X" as an example.
    """
    out = []
    for row in xyz_np:
        x, y, z = row
        out.append(["X", str(x), str(y), str(z)])
    return out

def flatten_geometry(xyz_lines):
    """
    Flatten xyz-block lines into a list.
    """
    flat = []
    for line in xyz_lines:
        _, coords = parse_xyz_line(line)
        flat.extend(coords)
    return flat

def dot_product(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2))

def parse_normal_modes(hess_filename):
    """
    Parse the $normal_modes block in the .hess file.
    Returns a list of normal mode vectors.
    """
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
    """
    Extract the q indices from the title string as 0-based indices.
    Example: "q1" -> [0], "q1q2" -> [0, 1].
    """
    qs = re.findall(r"q(\d+)", grid_title)
    return [int(q) - 1 for q in qs]

def parse_atom_masses_from_hess(hess_filename):
    """
    Parse the $atoms block in the .hess file to get the number of atoms and their masses.
    """
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
    """
    Take the difference between coords and eq_coords (both flattened),
    multiply each atomic displacement by sqrt(mass), and return the weighted displacement vector.
    """
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

# --- 以下、グリッド順序用の関数群 ---
def index_to_multi(index, G, N):
    """
    与えられた grid_index (整数) を、G進数で N 桁のタプル (i1, i2, ..., iN) に変換する。
    下位桁が q_N として扱うため、最終的に逆順にして返す。
    """
    digits = []
    for _ in range(N):
        digits.append(index % G)
        index //= G
    return tuple(reversed(digits))

def generate_multi_indices(G, N):
    """
    すべての multi-index タプル (i1, i2, ..., iN) （各 i_j は 0～G-1）
    を生成する。出力順序は q_N, q_(N-1), …, q1 の順となるように、タプルの逆順でソートする。
    """
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

def fix_tasks_array(tasks_list, G, N, masses, eq_geometry, normal_modes, offset, tol=1e-6):
    """
    tasks_list: multi-index順に並んだタスク（1Dリスト）
    G: 各軸のグリッド数
    N: qの次元数
    masses, eq_geometry, normal_modes, offset: 既存のパラメータ
    tol: 判定用の閾値
    
    まず各タスクに対して，q値を計算して task["q_vals"] として [q1, q2, ..., qN]（q1が内側）を保存する。
    その後，タスクリストを (G,)*N の配列にreshapeし，各内側軸（すなわち，array軸 1～N-1，対応する q が q1～q_(N-1)）
    の中心（index = (G-1)//2）のタスクで，q値がほぼ0なら，直前のサイクルの値をコピーする。
    最終的に修正済みのリストを返す。
    """
    arr = np.array(tasks_list, dtype=object).reshape((G,)*N)
    center = (G - 1) // 2
    # 各タスクに q_vals を計算して保存
    for idx in np.ndindex(arr.shape):
        task = arr[idx]
        disp = mass_weighted_displacement(task["grid_flat"], eq_geometry, masses)
        # 計算順は，仮にarray軸0が q_N, 軸1が q_(N-1), ..., 軸N-1が q1 とする
        q_computed = []
        for d in range(N):
            q_val = 50.0 * dot_product(disp, normal_modes[offset + d])
            q_computed.append(q_val)
        # task["q_vals"] を [q1, q2, ..., qN]（逆順）として保存
        task["q_vals"] = list(reversed(q_computed))
    # 内側の各軸（array軸 1～N-1，対応する mode: q_(N-1),..., q1）について修正
    # すなわち，for each axis a in 1...N-1, for each fixed index in other軸, if the q値 for mode (a) is nearly 0,
    # then copy the value from index (center - 1) along axis a.
    for a in range(1, N):
        mode_index = a  # mode_index in q_vals: 0 -> q1, 1 -> q2, etc.
        # iterate over all indices in arr
        for idx in np.ndindex(arr.shape):
            if idx[a] == center:
                task = arr[idx]
                if abs(task["q_vals"][mode_index]) < tol:
                    # copy from same index except along axis a, use center-1
                    idx_prev = list(idx)
                    idx_prev[a] = center - 1
                    idx_prev = tuple(idx_prev)
                    task["q_vals"][mode_index] = arr[idx_prev]["q_vals"][mode_index]
    fixed_list = arr.flatten().tolist()
    return fixed_list
def reorder_and_fill_points(tasks, G, N, grid_title, eq_geometry):
    """
    tasks: 各グリッドブロック（辞書）のリスト（grid_index キーあり）
    G: 各軸のグリッド数
    N: 次元数（qの数）
    grid_title: 例 "q3q2q1" など
    eq_geometry: 補完用の平衡座標

    flat な grid_index（整数）をキーにして、全 G^N 個のスロットに対応付けます。
    取得されなかった箇所は dummy として平衡座標、0差分で埋めます。
    """
    # tasks から grid_index をキーに辞書を作成
    indexed_tasks = {t["grid_index"]: t for t in tasks if t["grid_index"] is not None}
    # すべての multi-index（タプル）を生成し、1D 用の flat index を計算
    all_indices = generate_multi_indices(G, N)
    ordered_tasks = []
    for mi in all_indices:
        flat_index = sum(mi[i] * (G**(N-i-1)) for i in range(N))
        if flat_index in indexed_tasks:
            ordered_tasks.append(indexed_tasks[flat_index])
        else:
            ordered_tasks.append({
                "grid_title": grid_title,
                "num": G,
                "grid_index": flat_index,
                "grid_flat": eq_geometry,
                "calc_id": "dummy",
                "energy_diff": 0.0,
                "dipole_diff": (0.0, 0.0, 0.0)
            })
    return ordered_tasks
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
        "energy_diff": energy - eq_energy,  # eq_energy は main() 内で定義済み
        "dipole_diff": (dipole[0] - eq_dipole[0],
                        dipole[1] - eq_dipole[1],
                        dipole[2] - eq_dipole[2])
    }

# ---------------------------
# Main
# ---------------------------
def main():
    global eq_energy, eq_dipole  # run_orca_job 内で利用するため
    with open(xyz_filename, 'r') as f:
        lines = f.readlines()
    i = 0
    nlines = len(lines)
    calc_counter = 0
    pot_data_by_title = {}    # キーは grid_title、値はタスク辞書のリスト
    dipole_data_by_title = {}

    # --- Equilibrium geometry の読み込み ---
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

    grid_tasks = []
    # --- 読み込んだ makeGrid.xyz から各 mkg- ブロックを抽出 ---
    while i < nlines:
        if not lines[i].strip().isdigit():
            i += 1
            continue
        natoms = int(lines[i].strip())
        title_line = lines[i+1].strip()
        # タイトルは "mkg-<q情報>-<グリッド数>-<グリッド内番号>" の形式とする
        remainder = title_line[len("mkg-"):]
        parts = remainder.split("-")
        grid_title = parts[0]         # 例: "q2q1" または "q3q2q1" など
        num = int(parts[1])           # グリッド数 G（例: 11）
        grid_index = int(parts[2]) if len(parts) >= 3 else None
        grid_xyz_lines = lines[i+2 : i+2+natoms]
        grid_tasks.append({
            "grid_title": grid_title,
            "num": num,
            "grid_index": grid_index,
            "grid_xyz_lines": grid_xyz_lines,
            "calc_id": str(calc_counter)
        })
        calc_counter += 1
        i += 2 + natoms

    # --- ORCA 計算の並列実行 ---
    with ThreadPoolExecutor() as executor:
        future_to_task = {}
        for task in grid_tasks:
            future = executor.submit(run_orca_job,
                                     task["grid_xyz_lines"],
                                     task["calc_id"],
                                     task["grid_title"],
                                     task["num"],
                                     task["grid_index"])
            future_to_task[future] = task
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"calc_{task['calc_id']} ({task['grid_title']}) generated an exception: {exc}")
                continue
            grid_title = result["grid_title"]
            if grid_title not in pot_data_by_title:
                pot_data_by_title[grid_title] = {"num": result["num"], "tasks": []}
                dipole_data_by_title[grid_title] = {"num": result["num"], "tasks": []}
            pot_data_by_title[grid_title]["tasks"].append(result)
            dipole_data_by_title[grid_title]["tasks"].append(result)
            print(f"calc_{result['calc_id']} ({grid_title}): ΔE = {result['energy_diff']}, ΔDipole = {result['dipole_diff']}")

    # --- Parse normal modes and masses from Hess file ---
    normal_modes = parse_normal_modes(hess_filename)
    if is_line:
        offset = 5   # For linear molecules
    else:
        offset = 6   # For non-linear molecules
    _, masses = parse_atom_masses_from_hess(hess_filename)

    # --- 各グリッドグループについて、multi-index で順序付けと dummy 補完 ---
    for grid_title, group in pot_data_by_title.items():
        q_count = len(re.findall(r"q\d+", grid_title))
        G = group["num"]
        ordered_tasks = reorder_and_fill_points(group["tasks"], G, q_count, grid_title, eq_geometry)
        # ここで、ordered_tasks は1Dリスト（長さ G^q_count）となっているので，
        # fix_tasks_array() を呼び出して内部軸（q₁～qₙ₋₁）の中心での dummy を修正する
        fixed_tasks = fix_tasks_array(ordered_tasks, G, q_count, masses, eq_geometry, normal_modes, offset)
        pot_data_by_title[grid_title]["ordered"] = fixed_tasks
    for grid_title, group in dipole_data_by_title.items():
        q_count = len(re.findall(r"q\d+", grid_title))
        G = group["num"]
        ordered_tasks = reorder_and_fill_points(group["tasks"], G, q_count, grid_title, eq_geometry)
        fixed_tasks = fix_tasks_array(ordered_tasks, G, q_count, masses, eq_geometry, normal_modes, offset)
        dipole_data_by_title[grid_title]["ordered"] = fixed_tasks

    # --- Create pot files (energy differences) ---
    for grid_title, data in pot_data_by_title.items():
        num = data["num"]
        tasks_ordered = data["ordered"]
        # mode_names を [q1, q2, ..., qN]（昇順）とする
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
            for task in tasks_ordered:
                # task["q_vals"] は [q1, q2, ..., qN]（q1が最初）
                q_vals = task.get("q_vals", [50.0 * dot_product(mass_weighted_displacement(task["grid_flat"], eq_geometry, masses), normal_modes[offset + i]) for i in range(q_count)])
                # 出力は q1, q2, ..., qN とする
                line_str = "     ".join([f"{q:16.8f}" for q in q_vals] +
                                         [f"{task['energy_diff']:16.10e}"])
                fout.write(line_str + "\n")
        print(f"Created: {out_filename}  (Data points: {len(tasks_ordered)})")

    # --- Create dipole files ---
    for grid_title, data in dipole_data_by_title.items():
        num = data["num"]
        tasks_ordered = data["ordered"]
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
            for task in tasks_ordered:
                q_vals = task.get("q_vals", [50.0 * dot_product(mass_weighted_displacement(task["grid_flat"], eq_geometry, masses), normal_modes[offset + i]) for i in range(q_count)])
                line_str = "     ".join([f"{q:16.8f}" for q in q_vals] +
                                        [f"{task['dipole_diff'][0]:16.10e}",
                                         f"{task['dipole_diff'][1]:16.10e}",
                                         f"{task['dipole_diff'][2]:16.10e}"])
                fout.write(line_str + "\n")
        print(f"Created: {out_filename}  (Data points: {len(tasks_ordered)})")

    # --- Create q0.pot file for the equilibrium geometry ---
    q0_filename = os.path.join(output_dir, "q0.pot")
    with open(q0_filename, "w") as fout:
        fout.write("# Number of data\n")
        fout.write("     1 \n")
        fout.write("# Data at the reference geometry\n")
        fout.write(f"  {eq_energy:14.8e}\n")
    print(f"Created: {q0_filename} (Equilibrium energy {eq_energy})")

    # --- Create q0.dipole file for the equilibrium geometry ---
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

