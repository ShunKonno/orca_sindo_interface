import glob
import os
import xml.etree.ElementTree as ET
def create_xml(minfo_files):
    root = ET.Element("makePES")
    for minfo in minfo_files:
        minfo_elem = ET.SubElement(root, "minfoFile", value=f"{minfo}.minfo")

        ET.SubElement(root, "MR", value="3")
        ET.SubElement(root, "dipole", value="true")

        qchem = ET.SubElement(root, "qchem")
        ET.SubElement(qchem, "program", value="generic")
        ET.SubElement(qchem, "title", value="B3LYP/cc-pVDZ")
        ET.SubElement(qchem, "xyzfile", value="makeGrid")

        grid = ET.SubElement(root, "grid")
        ET.SubElement(grid, "ngrid", value="11")
        ET.SubElement(grid, "fullmc", value="true")

        tree = ET.ElementTree(root)
        ET.indent(tree, space="   ")
        tree.write(f"../output/{minfo}_grid.xml", encoding="UTF-8", xml_declaration=True)

def create_run_script(minfo_files, output_file="../output/run_grid.sh"):
    script_content = """#!/bin/bash

. /home/shun/SINDO/sindo/sindovars.sh
"""
    for minfo in minfo_files:
        xml_filename = f"{minfo}_grid.xml"
        script_content += f"java RunMakePES -f {xml_filename} >& makeGRID.out\n"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(script_content)

def main():
    def get_minfo_files(directory):
        return [os.path.splitext(os.path.basename(file))[0] for file in glob.glob(f"{directory}/*.minfo")]
    
    directory_path = "../output"
    minfo_files = get_minfo_files(directory_path)

    create_xml(minfo_files)
    create_run_script(minfo_files)


if __name__  == "__main__":
    main()
    print("Complete!")
