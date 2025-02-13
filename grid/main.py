#!/usr/bin/env python3
import subprocess
import re
import os

# ---------------------------
# 設定
# ---------------------------
# ORCA計算用ヘッダ（※inpファイルの基底関数等は実際にはinpファイル先頭部から取得してください）
orca_header = """! B3LYP def2-SVP TightSCF Freq

* xyz 0 1
"""

# ここではpotファイルの先頭行に出力する基底関数情報をハードコード（例）
basis_info = "B3LYP/def2-SVP"

# hessファイル名（平衡状態計算用）
hess_filename = "calc_eq.hess"

# 入力xyzファイル名
xyz_filename = "makeGrid.xyz"

# 出力先ディレクトリ
output_dir = os.path.join("..", "output", "pot")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# 補助関数
# ---------------------------
def extract_energy(output_file):
    """
    ORCAの出力ファイルから "FINAL SINGLE POINT ENERGY" 行からエネルギー値を抽出
    """
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
    """
    座標リスト（各行の文字列）から ORCA 入力ファイルを作成し計算実行する関数。
    calc_id は出力ファイル名の識別子として使用。
    平衡状態の場合（calc_id=="eq"）は .out および .hess ファイルを削除せず保持する。
    それ以外は、エネルギー取得後に一時ファイル（inp, out, hess）を削除する。
    """
    input_filename = f"calc_{calc_id}.inp"
    output_filename = f"calc_{calc_id}.out"

    # 入力ファイル作成（XYZセクションは座標行の連結＋最後に "*" で閉じる）
    input_content = orca_header + "".join(coordinates) + "*\n"
    with open(input_filename, 'w') as f:
        f.write(input_content)

    try:
        subprocess.run(["orca", input_filename], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error in ORCA calculation for {input_filename}:")
        print(e.stderr.decode())
        return None

    energy = extract_energy(output_filename)

    if calc_id != "eq":
        try:
            os.remove(input_filename)
            os.remove(output_filename)
            # ORCAが出力する場合の .hess ファイルも削除（calc_eq.hess は平衡状態用なので、ここは削除しない）
            hess_file = f"calc_{calc_id}.hess"
            if os.path.exists(hess_file):
                os.remove(hess_file)
        except Exception as e:
            print("Error deleting temporary files:", e)

    return energy

def parse_xyz_line(line):
    """
    xyzフォーマットの1行をパースして、原子シンボルと3個の座標(float)を返す。
    例: "   N         -0.0000051847          0.0000000054          0.0826483076"
    """
    tokens = line.split()
    if len(tokens) < 4:
        raise ValueError("XYZ行のフォーマットが不正です:" + line)
    try:
        coords = [float(tok) for tok in tokens[1:4]]
    except Exception as e:
        raise ValueError(f"座標の変換エラー: {line}\n{e}")
    return tokens[0], coords

def flatten_geometry(xyz_lines):
    """
    xyzブロックの各行から座標をパースし、1次元リスト（[x1, y1, z1, x2, y2, z2, ...]）として返す。
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
    hessファイル内の "$normal_modes" ブロックから正規振動モードの行列をパースする。
    入力例では、最初の1～2行にサイズ情報があり、その後ブロックごとに列番号のヘッダーと数値データが続く形式を想定。
    戻り値は、各モードを1次元リスト（長さ 3N）としたリスト（モード番号順）。
    """
    with open(hess_filename, 'r') as f:
        lines = f.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("$normal_modes"):
            start = i
            break
    if start is None:
        raise ValueError("hessファイル内に$normal_modesブロックが見つかりません")

    dims_line = lines[start+1].strip()
    dims = dims_line.split()
    nrows = int(dims[0])
    ncols = int(dims[1])

    # 初期化：matrix[row][col]
    matrix = [[None] * ncols for _ in range(nrows)]

    line_index = start + 2
    cols_processed = 0
    while cols_processed < ncols:
        # スキップ：空行など
        while line_index < len(lines) and not lines[line_index].strip():
            line_index += 1
        header_line = lines[line_index].strip()
        block_cols = []
        for tok in header_line.split():
            try:
                block_cols.append(int(tok))
            except:
                pass
        block_ncols = len(block_cols)
        line_index += 1
        # 各行について、行番号の後に block_ncols 個の値が記載される
        for row in range(nrows):
            row_line = lines[line_index].strip()
            line_index += 1
            tokens = row_line.split()
            values = [float(x) for x in tokens[1:]]
            for j, col in enumerate(block_cols):
                matrix[row][col] = values[j]
        cols_processed += block_ncols

    # 各モードは各列
    modes = []
    for col in range(ncols):
        mode = [matrix[row][col] for row in range(nrows)]
        modes.append(mode)
    return modes

def extract_mode_indices(grid_title):
    """
    タイトル文字列（例："q1" や "q1q2" や "q1q2q3"）から正規振動モード番号を抽出する。
    ここでは "q1" → 1, "q2" → 2, … として、計算時はインデックスとして0引きにする。
    """
    qs = re.findall(r"q(\d+)", grid_title)
    return [int(q) - 1 for q in qs]  # 0-indexed

# ---------------------------
# メイン処理
# ---------------------------
def main():
    with open(xyz_filename, 'r') as f:
        lines = f.readlines()

    i = 0
    nlines = len(lines)
    calc_counter = 0

    eq_energy = None
    eq_geometry = None  # 平衡状態の座標（flattened）

    # pot_data_by_title: { grid_title: { "num": num, "points": [(grid_flat, energy_diff), ...] } }
    pot_data_by_title = {}

    # --- 最初のブロックは平衡状態（タイトル "mkg-eq"） ---
    if i < nlines:
        try:
            natoms = int(lines[i].strip())
        except Exception as e:
            print("最初の行が整数になっていません。")
            return
        eq_title = lines[i+1].strip()
        if eq_title != "mkg-eq":
            print("最初のブロックは平衡状態 (mkg-eq) ではありません。")
            return
        eq_xyz_lines = lines[i+2:i+2+natoms]
        eq_geometry = flatten_geometry(eq_xyz_lines)
        # 平衡状態の計算は calc_id="eq" として実行 → .out, .hess は保持
        eq_energy = run_orca_calculation(eq_xyz_lines, "eq")
        if eq_energy is None:
            print("平衡状態のエネルギー取得に失敗しました。")
            return
        print(f"平衡状態エネルギー: {eq_energy}")
        i += 2 + natoms
        calc_counter += 1
    else:
        print("入力ファイルが空です。")
        return

    # --- 残りのブロック（各グリッド点）を処理 ---
    while i < nlines:
        line = lines[i].strip()
        if not line.isdigit():
            i += 1
            continue
        natoms = int(line)
        title_line = lines[i+1].strip()  # 例："mkg-q1-11-0" や "mkg-q1q2-11-45" など
        if not title_line.startswith("mkg-"):
            print(f"予期しないタイトル形式: {title_line}")
            i += 2 + natoms
            continue

        # タイトル形式 "mkg-{grid_title}-{num}-{...}"
        remainder = title_line[len("mkg-"):]
        parts = remainder.split("-")
        if len(parts) < 3:
            print(f"タイトルの形式が不正です: {title_line}")
            i += 2 + natoms
            continue

        grid_title = parts[0]   # 例："q1" または "q1q2" etc.
        num_str = parts[1]
        try:
            num = int(num_str)
        except ValueError:
            print(f"タイトルから数値抽出失敗: {title_line}")
            i += 2 + natoms
            continue

        grid_xyz_lines = lines[i+2:i+2+natoms]
        grid_flat = flatten_geometry(grid_xyz_lines)

        energy = run_orca_calculation(grid_xyz_lines, str(calc_counter))
        if energy is None:
            print(f"計算 {calc_counter} でエラー発生")
            i += 2 + natoms
            calc_counter += 1
            continue

        energy_diff = energy - eq_energy

        if grid_title not in pot_data_by_title:
            pot_data_by_title[grid_title] = {"num": num, "points": []}
        pot_data_by_title[grid_title]["points"].append((grid_flat, energy_diff))

        print(f"計算 {calc_counter} ({grid_title}): エネルギー差 = {energy_diff}")
        calc_counter += 1
        i += 2 + natoms

    # --- 正規振動モードをhessファイルからパース ---
    try:
        normal_modes = parse_normal_modes(hess_filename)
    except Exception as e:
        print("正規振動モードのパースに失敗しました。")
        print(e)
        return

    # --- 各タイトルごとにpotファイルを作成 ---
    for grid_title, data in pot_data_by_title.items():
        num = data["num"]
        points = data["points"]
        mode_indices = extract_mode_indices(grid_title)
        n_modes = len(mode_indices)
        expected_points = (num ** n_modes) - 1 if n_modes > 0 else 0
        if expected_points and len(points) != expected_points:
            print(f"警告: タイトル {grid_title} の点数が期待値 {expected_points} と異なります（実際: {len(points)}）")

        out_filename = os.path.join(output_dir, f"{grid_title}.pot")
        with open(out_filename, "w") as fout:
            # 1行目: 基底関数情報と、タイトルから取得した数値（qが n_modes 個の場合は "(num num ...)"）
            num_list_str = " ".join([str(num)] * n_modes)
            fout.write(f"{basis_info} ({num_list_str})\n")

            # 2行目: ヘッダー "# Number of grids and data"
            fout.write("# Number of grids and data\n")
            # 3行目: 各軸のグリッド数＋最後に 1（例: q1なら "      11      1", q1q2なら "      11      11      1"）
            grid_dims = ("".join([f"{num:8d}" for _ in range(n_modes)]) + f"{1:8d}")
            fout.write(grid_dims + "\n")

            # 4行目: カラム見出し（例: "#   q1              Energy" または "#   q1              q2              Energy"）
            header_line = "#"
            for mi in range(n_modes):
                header_line += f"   q{mi+1:<14s}"
            header_line += "  Energy"
            fout.write(header_line + "\n")

            # 各グリッド点ごとに、平衡状態との差から変位ベクトルを計算し、指定正規振動モードとの内積（＝各q座標）を算出
            for (grid_flat, energy_diff) in points:
                # 変位ベクトル（各座標成分: grid - eq）
                disp = [g - e for g, e in zip(grid_flat, eq_geometry)]
                q_values = []
                for mode_idx in mode_indices:
                    # 今回は原子重み付けなしの単純な内積
                    proj = dot_product(disp, normal_modes[mode_idx])
                    q_values.append(proj)
                # 各q値を16桁浮動小数点形式、エネルギーは指数表記
                q_str = "".join([f"{q:16.8f}" for q in q_values])
                energy_str = f"{energy_diff:16.10e}"
                fout.write(q_str + energy_str + "\n")
        print(f"タイトル {grid_title} のpotデータ（{len(points)}点）を {out_filename} に保存しました")

if __name__ == "__main__":
    main()
