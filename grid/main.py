#!/usr/bin/env python3
import subprocess
import re
import os
import math

# ---------------------------
# 設定
# ---------------------------
# 平衡状態計算（最初の1回）用のORCA入力ヘッダ：Freq付き
orca_header_eq = """! B3LYP def2-SVP TightSCF Freq
%pal nprocs 21 end
* xyz 0 1
"""

# それ以外（エネルギー計算用）のORCA入力ヘッダ：Freqなし
orca_header_sp = """! B3LYP def2-SVP TightSCF
%pal nprocs 21 end
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
    """ORCA出力ファイルからFINAL SINGLE POINT ENERGYを抽出"""
    energy = None
    with open(output_file, 'r') as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                m = re.search(r"FINAL SINGLE POINT ENERGY\s+([-+]?\d*\.\d+|\d+)", line)
                if m:
                    energy = float(m.group(1))
                    break
    return energy

def run_orca_calculation(coordinates, calc_id):
    """coordinates(文字列リスト)からORCA入力ファイルを作成し計算実行"""
    if calc_id == "eq":
        orca_header = orca_header_eq
    else:
        orca_header = orca_header_sp

    input_filename = f"calc_{calc_id}.inp"
    output_filename = f"calc_{calc_id}.out"

    input_content = orca_header + "".join(coordinates) + "*\n"
    with open(input_filename, 'w') as f:
        f.write(input_content)

    orca_path = "/home/shun/orca/orca"
    cmd = f"{orca_path} {input_filename} > {output_filename}"
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in ORCA calculation for {input_filename}: {e}")
        return None

    energy = extract_energy(output_filename)

    # freqなし計算はファイルを削除
    if calc_id != "eq":
        for fn in (input_filename, output_filename, f"calc_{calc_id}.hess"):
            if os.path.exists(fn):
                os.remove(fn)

    return energy

def parse_xyz_line(line):
    """xyzフォーマット1行をパースして(原子名, (x,y,z))を返す"""
    tokens = line.split()
    if len(tokens) < 4:
        raise ValueError("XYZ行の形式が不正: " + line)
    atom = tokens[0]
    coords = list(map(float, tokens[1:4]))
    return atom, coords

def flatten_geometry(xyz_lines):
    """xyzブロックを平坦化して [x1,y1,z1,x2,y2,z2,...] の配列を返す"""
    flat = []
    for line in xyz_lines:
        _, c = parse_xyz_line(line)
        flat.extend(c)
    return flat

def dot_product(vec1, vec2):
    return sum(a*b for a, b in zip(vec1, vec2))

def parse_normal_modes(hess_filename):
    """
    $normal_modesブロックをパースし、列方向に各モードのベクトル(長さ3N)を格納した
    リストを返す (modes[col] = mode_vector)。
    """
    with open(hess_filename, 'r') as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("$normal_modes"):
            start = i
            break
    if start is None:
        raise ValueError("hessファイルに $normal_modes ブロックが見つかりません")

    dims_line = lines[start+1].strip()
    dims = dims_line.split()
    nrows = int(dims[0])
    ncols = int(dims[1])

    matrix = [[None]*ncols for _ in range(nrows)]
    line_index = start + 2
    cols_processed = 0

    while cols_processed < ncols:
        # 空行をスキップ
        while line_index < len(lines) and not lines[line_index].strip():
            line_index += 1
        # ブロックヘッダ行 (例: "          0           1   ...")
        header_line = lines[line_index].strip()
        block_cols = []
        for tok in header_line.split():
            try:
                block_cols.append(int(tok))
            except:
                pass
        block_ncols = len(block_cols)
        line_index += 1

        # 各行のデータ
        for row in range(nrows):
            row_line = lines[line_index].strip()
            line_index += 1
            tokens = row_line.split()
            # tokens[0] は行番号。それ以降が数値
            values = list(map(float, tokens[1:]))
            for j, col in enumerate(block_cols):
                matrix[row][col] = values[j]

        cols_processed += block_ncols

    # 列方向にモードベクトルをまとめる
    modes = []
    for col in range(ncols):
        mode_col = [matrix[row][col] for row in range(nrows)]
        modes.append(mode_col)
    return modes

def extract_mode_indices(grid_title):
    """"q1"→[0], "q1q2"→[0,1] のように0始まりで返す"""
    qs = re.findall(r"q(\d+)", grid_title)
    return [int(q)-1 for q in qs]

# --- Hessファイル中の $atoms ブロックから原子数・質量を取得 ---
def parse_atom_masses_from_hess(hess_filename):
    """
    Hessファイル中の $atoms ブロックから原子数と各原子の質量をリストとして返す。
    例:
      $atoms
      4
       N     14.00700     -0.000002319545     0.000000002416     0.072487142001
       H      1.00800      0.937813473569    -0.000000029521    -0.335759692919
       ...
    """
    with open(hess_filename, 'r') as f:
        lines = f.readlines()

    start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("$atoms"):
            start_index = i
            break
    if start_index is None:
        raise ValueError("$atoms ブロックが見つかりません。")

    n_atoms = int(lines[start_index+1].strip())
    masses = []
    for j in range(n_atoms):
        tokens = lines[start_index+2+j].split()
        mass = float(tokens[1])
        masses.append(mass)
    return n_atoms, masses

def mass_weighted_displacement(coords, eq_coords, masses):
    """
    平衡構造との座標差 (coords - eq_coords) に対し、各原子ごとに
    √mass を掛けた重み付け変位を返す。
    coords, eq_coords は [x1,y1,z1, x2,y2,z2, ...] の形式で与えられる。
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

    # --- 最初のブロック(平衡構造) ---
    if i < nlines:
        try:
            natoms = int(lines[i].strip())
        except Exception as e:
            print("最初の行が原子数(int)でありません。")
            return
        n_atoms = natoms
        eq_title_line = lines[i+1].strip()
        if eq_title_line != "mkg-eq":
            print("最初のブロックは mkg-eq ではありません。")
            return

        eq_xyz_lines = lines[i+2 : i+2+natoms]
        eq_geometry = flatten_geometry(eq_xyz_lines)

        eq_energy = run_orca_calculation(eq_xyz_lines, "eq")
        if eq_energy is None:
            print("平衡構造のエネルギー取得失敗")
            return
        print(f"平衡構造エネルギー: {eq_energy}")

        i += 2 + natoms
        calc_counter += 1
    else:
        print("入力ファイルが空です")
        return

    # --- 残りのブロック ---
    while i < nlines:
        line = lines[i].strip()
        if not line.isdigit():
            i += 1
            continue

        natoms = int(line)
        title_line = lines[i+1].strip()
        if not title_line.startswith("mkg-"):
            print(f"想定外のタイトル: {title_line}")
            i += 2 + natoms
            continue

        # タイトル例: "mkg-q1-1-..."
        remainder = title_line[len("mkg-"):]
        parts = remainder.split("-")
        if len(parts) < 3:
            print(f"タイトル形式が不正: {title_line}")
            i += 2 + natoms
            continue

        grid_title = parts[0]  # 例: "q1"
        num_str = parts[1]
        try:
            num = int(num_str)
        except:
            print(f"タイトルから数値取得失敗: {title_line}")
            i += 2 + natoms
            continue

        grid_xyz_lines = lines[i+2 : i+2+natoms]
        grid_flat = flatten_geometry(grid_xyz_lines)

        energy = run_orca_calculation(grid_xyz_lines, str(calc_counter))
        if energy is None:
            print(f"calc_{calc_counter} エネルギー取得失敗")
            i += 2 + natoms
            calc_counter += 1
            continue

        energy_diff = energy - eq_energy

        if grid_title not in pot_data_by_title:
            pot_data_by_title[grid_title] = {"num": num, "points": []}
        pot_data_by_title[grid_title]["points"].append((grid_flat, energy_diff))

        print(f"calc_{calc_counter} ({grid_title}): ΔE = {energy_diff}")
        calc_counter += 1
        i += 2 + natoms

    # --- 正規振動モードを読み込み ---
    try:
        normal_modes = parse_normal_modes(hess_filename)
    except Exception as e:
        print("正規振動モードパース失敗:", e)
        return

    # 直線分子用としてoffsetを固定
    offset = 6

    # --- Hessファイルから原子の質量を取得 ---
    try:
        atoms_count, masses = parse_atom_masses_from_hess(hess_filename)
    except Exception as e:
        print("原子の質量のパース失敗:", e)
        return
    if atoms_count != n_atoms:
        print("警告: Hessファイルの原子数とXYZファイルの原子数が一致しません。")

    # --- potファイル作成 ---
    for grid_title, data in pot_data_by_title.items():
        num = data["num"]   # 例: 11
        points = data["points"]
        mode_indices = extract_mode_indices(grid_title)  # 例: "q1"→[0]
        n_modes = len(mode_indices)
        # 今回は1次元想定なので n_modes=1 を期待

        # ===== 1) 中点 (q=0) を挿入して合計11データにする =====
        # 平衡構造とエネルギー差0をリストに追加
        points.append((eq_geometry, 0.0))

        # q値でソート（1次元の場合 mode_indices[0]のみ使用）
        if n_modes == 1:
            def get_q(p):
                # 質量重み付け変位を計算
                disp = mass_weighted_displacement(p[0], eq_geometry, masses)
                return dot_product(disp, normal_modes[offset + mode_indices[0]])
            points.sort(key=get_q)
        else:
            pass

        # ===== 2) 出力 (ヘッダと列の位置合わせ) =====
        out_filename = os.path.join(output_dir, f"{grid_title}.pot")
        with open(out_filename, "w") as fout:
            # 1行目
            num_list_str = " ".join(str(num) for _ in range(n_modes)) if n_modes > 0 else "0"
            fout.write(f"{basis_info} ({num_list_str})\n")
            # 2行目
            fout.write("# Number of grids and data\n")
            # 3行目
            grid_dims = "".join([f"{num:8d}" for _ in range(n_modes)]) + f"{1:8d}"
            fout.write(grid_dims + "\n")
            # 4行目: ヘッダー
            header_line = "#   q1              Energy"
            fout.write(header_line + "\n")
            # データ部
            for coords_flat, e_diff in points:
                disp = mass_weighted_displacement(coords_flat, eq_geometry, masses)
                if n_modes == 1:
                    q_val = 50.0*dot_product(disp, normal_modes[offset + mode_indices[0]])
                    fout.write(f"  {q_val:10.8f}      {e_diff:14.8e}\n")
                else:
                    pass
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

