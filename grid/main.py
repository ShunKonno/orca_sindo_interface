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
    """ORCA出力ファイルから FINAL SINGLE POINT ENERGY を抽出"""
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
    """ORCA出力ファイルから Total Dipole Moment を抽出し、(X,Y,Z) のタプルで返す"""
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
    coordinates（文字列リスト）から ORCA 入力ファイルを作成し計算実行する。
    calc_id=="eq" の場合は Freq 付きヘッダを使用し、その他は Freq なし。
    出力ファイルからエネルギーと Total Dipole Moment を抽出して返す。
    平衡状態以外の場合、計算後の一時ファイルは削除する。
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
    """xyzフォーマット1行をパースして (原子名, (x,y,z)) を返す"""
    tokens = line.split()
    atom = tokens[0]
    coords = list(map(float, tokens[1:4]))
    return atom, coords

def xyz_block_to_np(xyz_lines):
    """
    xyzフォーマットのブロック（複数行）から (N,3) の NumPy 配列を返す
    """
    coords_list = []
    for line in xyz_lines:
        _, coord = parse_xyz_line(line)
        coords_list.append(coord)
    return np.array(coords_list)  # shape = (N, 3)

def np_to_linecheck_format(xyz_np):
    # xyz_np: shape=(N,3)
    # 戻り値: [[symbol, x, y, z], [symbol, x, y, z], ...]
    # 例として全部 "X" を付与する
    out = []
    for row in xyz_np:
        x, y, z = row
        out.append(["X", str(x), str(y), str(z)])
    return out

def flatten_geometry(xyz_lines):
    """xyzブロックを [x1,y1,z1,x2,y2,z2,...] のリストに平坦化して返す"""
    flat = []
    for line in xyz_lines:
        _, coords = parse_xyz_line(line)
        flat.extend(coords)
    return flat

def dot_product(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2))

def parse_normal_modes(hess_filename):
    """
    hessファイル内の $normal_modes ブロックをパースし、
    各モード（列方向のベクトル、長さ3N）のリストを返す
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
    タイトル中の q 番号を抽出し、0始まりのリストとして返す。
    例: "q1" → [0], "q1q2" → [0, 1]
    """
    qs = re.findall(r"q(\d+)", grid_title)
    return [int(q) - 1 for q in qs]

def parse_atom_masses_from_hess(hess_filename):
    """
    Hessファイル中の $atoms ブロックから原子数と各原子の質量リストを返す
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
    coords と eq_coords（平坦化された座標リスト）の差に対して、各原子の √mass を掛けた重み付け変位ベクトルを返す
    """
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

# --- 並列実行用ジョブラッパー ---
def run_orca_job(grid_xyz_lines, calc_id, grid_title, num):
    energy, dipole = run_orca_calculation(grid_xyz_lines, calc_id)
    grid_flat = flatten_geometry(grid_xyz_lines)
    return (grid_title, num, grid_flat, energy, dipole)

# ---------------------------
# メイン処理
# ---------------------------
def main():
    # xyzファイル読み込み
    with open(xyz_filename, 'r') as f:
        lines = f.readlines()
    i = 0
    nlines = len(lines)
    calc_counter = 0
    eq_energy = None
    eq_dipole = None
    eq_geometry = None  # 平衡状態の座標（平坦化されたリスト）
    pot_data_by_title = {}      # エネルギー差用 { grid_title: {"num": num, "points": [(grid_flat, energy_diff), ...]} }
    dipole_data_by_title = {}   # 双極子差用 { grid_title: {"num": num, "points": [(grid_flat, dipole_diff), ...]} }

    # --- 平衡構造ブロック ---
    natoms = int(lines[i].strip())
    eq_title_line = lines[i+1].strip()
    eq_xyz_lines = lines[i+2 : i+2+natoms]

    # ここで np.array に変換して line_check に渡す
    eq_xyz_np = xyz_block_to_np(eq_xyz_lines)  # shape=(natoms,3)
    eq_xyz_for_linecheck = np_to_linecheck_format(eq_xyz_np)
    is_line = line_check(eq_xyz_for_linecheck)
    # 平坦化してリストに変換
    eq_geometry = eq_xyz_np.flatten().tolist()
    eq_energy, eq_dipole = run_orca_calculation(eq_xyz_lines, "eq")
    print(f"平衡構造エネルギー: {eq_energy}")
    print(f"平衡構造Dipole: {eq_dipole}")
    i += 2 + natoms
    calc_counter += 1

    # --- グリッド計算タスクの収集 ---
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

    # --- グリッド計算を並列実行 ---
    with ThreadPoolExecutor() as executor:
        future_to_task = {executor.submit(run_orca_job,
                          task["grid_xyz_lines"],
                          task["calc_id"],
                          task["grid_title"],
                          task["num"]): task for task in grid_tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                grid_title, num, grid_flat, energy, dipole = future.result()
            except Exception as exc:
                print(f"calc_{task['calc_id']} ({task['grid_title']}) generated an exception: {exc}")
                continue
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

    # --- Hessファイルから正規振動モードと原子質量を取得 ---
    normal_modes = parse_normal_modes(hess_filename)
    if is_line:
        offset = 5   # 直線分子の場合は6番目以降のモードを使用（0-indexで5）
    else:
        offset = 6   # 非直線分子の場合は7番目以降（0-indexで6）
    _, masses = parse_atom_masses_from_hess(hess_filename)

    # --- potファイル作成（エネルギー差） ---
    for grid_title, data in pot_data_by_title.items():
        num = data["num"]
        points = data["points"]
        mode_indices = extract_mode_indices(grid_title)  # 例: "q1"→[0], "q1q2"→[0,1] etc.
        n_modes = len(mode_indices)
        # ヘッダーは各列を16文字幅で出力（数値は全て16文字なので先頭位置が揃う）
        mode_names = re.findall(r"q\d+", grid_title)
        header_line = "#" + "".join([f"{name:16s}" for name in mode_names]) + f"{'Energy':16s}"
        # --- 基準状態 (全て0) を中間に挿入 ---
        if n_modes >= 1:
            points.sort(key=lambda p: 50.0 * dot_product(mass_weighted_displacement(p[0], eq_geometry, masses),
                                                           normal_modes[offset + mode_indices[0]]))
        mid = len(points) // 2
        points.insert(mid, (eq_geometry, 0.0))
        out_filename = os.path.join(output_dir, f"{grid_title}.pot")
        with open(out_filename, "w") as fout:
            fout.write(f"{basis_info} ({num})\n")
            fout.write("# Number of grids and data\n")
            grid_dims = "".join([f"{num:8d}" for _ in range(n_modes)]) + f"{1:8d}"
            fout.write(grid_dims + "\n")
            fout.write(header_line + "\n")
            for coords_flat, e_diff in points:
                disp = mass_weighted_displacement(coords_flat, eq_geometry, masses)
                q_values = []
                for idx in mode_indices:
                    q_val = 50.0 * dot_product(disp, normal_modes[offset + idx])
                    q_values.append(q_val)
                q_str = "".join([f"{q:16.8f}" for q in q_values])
                energy_str = f"{e_diff:16.10e}"
                fout.write(q_str +f"{'':4s}"+ energy_str + "\n")
        print(f"生成: {out_filename}  (データ数 {len(points)})")

    # --- dipoleファイル作成 ---
    for grid_title, data in dipole_data_by_title.items():
        num = data["num"]
        points = data["points"]
        mode_indices = extract_mode_indices(grid_title)
        n_modes = len(mode_indices)
        # 各列を16文字幅で出力
        mode_names = re.findall(r"q\d+", grid_title)
        header_line = "#" + "".join([f"{name:16s}" for name in mode_names]) + f"{'X':16s}{'Y':16s}{'Z':16s}"
        # --- 基準状態 (全0) を中間に挿入 ---
        if n_modes >= 1:
            points.sort(key=lambda p: 50.0 * dot_product(mass_weighted_displacement(p[0], eq_geometry, masses),
                                                           normal_modes[offset + mode_indices[0]]))
        mid = len(points) // 2
        points.insert(mid, (eq_geometry, (0.0, 0.0, 0.0)))
        out_filename = os.path.join(output_dir, f"{grid_title}.dipole")
        with open(out_filename, "w") as fout:
            fout.write(f"{basis_info}\n")
            fout.write("# Number of grids and data\n")
            grid_dims = "".join([f"{num:8d}" for _ in range(n_modes)]) + f"{3:8d}"
            fout.write(grid_dims + "\n")
            fout.write(header_line + "\n")
            for coords_flat, dip_diff in points:
                disp = mass_weighted_displacement(coords_flat, eq_geometry, masses)
                q_values = []
                for idx in mode_indices:
                    q_val = 50.0 * dot_product(disp, normal_modes[offset + idx])
                    q_values.append(q_val)
                q_str = "".join([f"{q:16.8f}" for q in q_values])
                dipole_str = f"{dip_diff[0]:20.10e}{dip_diff[1]:20.10e}{dip_diff[2]:20.10e}"
                fout.write(q_str + f"{'':4s}"+dipole_str + "\n")

        print(f"生成: {out_filename}  (データ数 {len(points)})")

    # --- 平衡状態のpotファイル (q0.pot) を出力 ---
    q0_filename = os.path.join(output_dir, "q0.pot")
    with open(q0_filename, "w") as fout:
        fout.write("# Number of data\n")
        fout.write("     1 \n")
        fout.write("# Data at the reference geometry\n")
        fout.write(f"  {eq_energy:14.8e}\n")
    print(f"生成: {q0_filename} (平衡状態のエネルギー {eq_energy})")

    # --- 平衡状態のdipoleファイル (q0.dipole) を出力 ---
    q0_dipole_filename = os.path.join(output_dir, "q0.dipole")
    with open(q0_dipole_filename, "w") as fout:
        fout.write("# Number of data\n")
        fout.write("     1 \n")
        fout.write("# Dipole at the reference geometry\n")
        fout.write(f"  {eq_dipole[0]:14.8e}  {eq_dipole[1]:14.8e}  {eq_dipole[2]:14.8e}\n")
    print(f"生成: {q0_dipole_filename} (平衡状態のDipole {eq_dipole})")

if __name__ == "__main__":
    main()
    for file in glob.glob("calc*"):
        os.remove(file)
    print("Complete!")

