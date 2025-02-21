#!/usr/bin/env python3

import os
import subprocess

def make_orca_input(block_name, coord_lines):
    charge = 0
    multiplicity = 1
    geom_block = "\n".join(coord_lines)
    orca_input = f"""! B3LYP def2-SVP TightSCF Freq Engrad
* xyz {charge} {multiplicity}
{geom_block}
*
%elprop
 Polar 1
end
"""
    return orca_input

def main():
    xyzfile = "../output/makeQFF.xyz"
    orca_cmd = "orca"
    os.makedirs("mkqff", exist_ok=True)
    with open(xyzfile, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    idx = 0
    total_lines = len(lines)

    while idx < total_lines:
        natom_line = lines[idx].strip()
        natom = int(natom_line)
        idx += 1

        block_name = lines[idx].strip()
        idx += 1

        coord_lines = lines[idx : idx + natom]
        idx += natom

        work_dir = f"mkqff/{block_name}"
        os.makedirs(work_dir, exist_ok=True)

        orca_input = make_orca_input(block_name, coord_lines)

        inp_path = os.path.join(work_dir, "job.inp")
        with open(inp_path, "w", encoding="utf-8") as f_inp:
            f_inp.write(orca_input)
        cmd = f"{orca_cmd} job.inp > job.out"
        print(f"[INFO] Running ORCA in {work_dir} ...")
        subprocess.run(cmd, shell=True, cwd=work_dir)

    print("All jobs submitted/finished.")

if __name__ == "__main__":
    main()
