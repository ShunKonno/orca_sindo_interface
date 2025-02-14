#!/usr/bin/env python3
import subprocess
import re
import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------
# 設定
# ---------------------------
# 平衡状態計算（最初の1回）用のORCA入力ヘッダ：Freq付き
orca_header_eq = """! B3LYP def2-SVP TightSCF Freq
* xyz 0 1
"""

# それ以外（エネルギー計算用）のORCA入力ヘッダ：Freqなし
orca_header_sp = """! B3LYP def2-SVP TightSCF
* xyz 0 1
"""

# potファイルの先頭行に出力する基底関数情報
basis_info = "B3LYP/def2-SVP"

# hessファイル名（平衡状態計算用に生成される .hess）
hess_filename = "calc_eq.hess"

# 入力xyzファイル名
xyz_filename = "../output/makeGrid.xyz"

# 出力先ディレクトリ
output_dir = os.path.join("..", "output", "pot")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# 補助関数
# ---------------------------
def extract_energy(output_file):
    # ORCA出力ファイルからFINAL SINGLE POINT ENERGYを抽出
    energy = None
    with open(output_file, 'r') as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                m = re.search(r"FINAL SINGLE POINT ENERGY\s+([-+]?\d*\.\d+|\d+)", line)
                energy = float(m.group(1))
                break
    return energy

def run_orca_calculation(coordinates, calc_id):
    # coordinates(文字列リスト)からORCA入力ファイルを作成し計算実行
    if calc_id == "eq":
        header = orca_header_eq
    else:
        header = orca_header_sp

    input_filename = f"calc_{calc_id}.inp"
    output_filename = f"calc_{calc_id}.out"
    input_content = header + "".join(coordinates) + "*\n"
    with open(input_filename, 'w') as f:
        f.write(input_content)
    orca_path = "orca"
    cmd = f"{orca_path} {input_filename} > {output_filename}"
    subprocess.run(cmd, shell=True, check=True)
    energy = extract_energy(output_filename)
    if calc_id != "eq":
        for fn in (input_filename, output_filename, f"calc_{calc_id}.hess"):
            if os.path.exists(fn):
                os.remove(fn)
    return energy

def parse_xyz_line(line):
    # xyzフォーマット1行をパースして(原子名, (x,y,z))を返す
    tokens = line.split()
    atom = tokens[0]
    coords = list(map(float, tokens[1:4]))
    return atom, coords

def flatten_geometry(xyz_lines):
    # xyzブロックを平坦化して [x1,y1,z1,x2,y2,z2,...] の配列を返す
    flat = []
    for line in xyz_lines:
        _, c = parse_xyz_line(line)
        flat.extend(c)
    return flat

def dot_product(vec1, vec2):
    return sum(a*b for a, b in zip(vec1, vec2))

def parse_normal_modes(hess_filename):
    # $normal_modesブロックをパースし、列方向に各モードのベクトル(長さ3N)を返す
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
    # "q1"→[0], "q1q2"→[0,1] のように0始まりで返す
    qs = re.findall(r"q(\d+)", grid_title)
    return [int(q)-1 for q in qs]

def parse_atom_masses_from_hess(hess_filename):
    # Hessファイル中の $atoms ブロックから原子数と各原子の質量を返す
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
    # 平衡構造との座標差 (coords - eq_coords) に対し、各原子ごとに√massを掛けた重み付け変位を返す
    if len(coords) != len(eq_coords):
        raise ValueError("座標のサイズが一致しません。")
    disp = []
    for i in range(0, len(coords), 3):
        dx = coords[i]   - eq_coords[i]
        dy = coords[i+1] - eq_coords[i+1]
        dz = coords[i+2] - eq_coords[i+2]
        m = masses[i // 3]
        disp.extend([dx * math.sqrt(m), dy * math.sqrt(m), dz * math.sqrt(m)])
    return disp

# ---------------------------
# メイン
# ---------------------------
def main():
    # xyzファイル読み込み
    with open(xyz_filename, 'r') as f:
        lines = f.readlines()
    i = 0
    nlines = len(lines)
    calc_counter = 0
    eq_energy = None
    eq_geometry = None
    n_atoms = None
    pot_data_by_title = {}

    # --- 平衡構造ブロック ---
    natoms = int(lines[i].strip())
    n_atoms = natoms
    eq_title_line = lines[i+1].strip()
    eq_xyz_lines = lines[i+2 : i+2+natoms]
    eq_geometry = flatten_geometry(eq_xyz_lines)
    eq_energy = run_orca_calculation(eq_xyz_lines, "eq")
    print(f"平衡構造エネルギー: {eq_energy}")
    i += 2 + natoms
    calc_counter += 1

    # --- グリッド計算タスクを収集 ---
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
        grid_flat = flatten_geometry(grid_xyz_lines)
        grid_tasks.append({
            "grid_title": grid_title,
            "num": num,
            "grid_xyz_lines": grid_xyz_lines,
            "grid_flat": grid_flat,
            "calc_id": str(calc_counter)
        })
        calc_counter += 1
        i += 2 + natoms

    # --- グリッド計算を並列実行 ---
    with ThreadPoolExecutor() as executor:
        future_to_task = {executor.submit(run_orca_calculation, task["grid_xyz_lines"], task["calc_id"]): task for task in grid_tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            energy = future.result()
            energy_diff = energy - eq_energy
            grid_title = task["grid_title"]
            if grid_title not in pot_data_by_title:
                pot_data_by_title[grid_title] = {"num": task["num"], "points": []}
            pot_data_by_title[grid_title]["points"].append((task["grid_flat"], energy_diff))
            print(f"calc_{task['calc_id']} ({grid_title}): ΔE = {energy_diff}")

    normal_modes = parse_normal_modes(hess_filename)
    offset = 6  # 直線分子用に固定
    atoms_count, masses = parse_atom_masses_from_hess(hess_filename)

    # --- 各グリッドのpotファイル作成 ---
    for grid_title, data in pot_data_by_title.items():
        num = data["num"]
        points = data["points"]
        mode_indices = extract_mode_indices(grid_title)
        n_modes = len(mode_indices)
        # 中点 (q=0) を追加
        points.append((eq_geometry, 0.0))
        if n_modes == 1:
            def get_q(p):
                disp = mass_weighted_displacement(p[0], eq_geometry, masses)
                return dot_product(disp, normal_modes[offset + mode_indices[0]])
            points.sort(key=get_q)
        out_filename = os.path.join(output_dir, f"{grid_title}.pot")
        with open(out_filename, "w") as fout:
            num_list_str = " ".join(str(num) for _ in range(n_modes)) if n_modes > 0 else "0"
            fout.write(f"{basis_info} ({num_list_str})\n")
            fout.write("# Number of grids and data\n")
            grid_dims = "".join([f"{num:8d}" for _ in range(n_modes)]) + f"{1:8d}"
            fout.write(grid_dims + "\n")
            header_line = "#   q1              Energy"
            fout.write(header_line + "\n")
            for coords_flat, e_diff in points:
                disp = mass_weighted_displacement(coords_flat, eq_geometry, masses)
                if n_modes == 1:
                    q_val = 50.0 * dot_product(disp, normal_modes[offset + mode_indices[0]])
                    fout.write(f"  {q_val:10.8f}      {e_diff:14.8e}\n")
        print(f"生成: {out_filename}  (データ数 {len(points)})")

    # --- 平衡状態のpotファイル (q0.pot) を出力 ---
    q0_filename = os.path.join(output_dir, "q0.pot")
    with open(q0_filename, "w") as fout:
        fout.write("# Number of data\n")
        fout.write("     1 \n")
        fout.write("# Data at the reference geometry\n")
        fout.write(f"  {eq_energy:14.8e}\n")
    print(f"生成: {q0_filename} (平衡状態のエネルギー {eq_energy})")

if __name__ == "__main__":
    main()

