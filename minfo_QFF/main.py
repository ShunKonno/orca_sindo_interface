import glob
import os
import xml.etree.ElementTree as ET
def create_xml(minfo_files):
    root = ET.Element("makePES")
    for minfo in minfo_files:
        minfo_elem = ET.SubElement(root, "minfoFile", value=f"{minfo}.minfo")

        ET.SubElement(root, "MR", value="3")

        qchem = ET.SubElement(root, "qchem")
        ET.SubElement(qchem, "program", value="generic")
        ET.SubElement(qchem, "title", value="B3LYP/cc-pVDZ")
        ET.SubElement(qchem, "xyzfile", value="makeQFF")

        qff = ET.SubElement(root, "qff")
        ET.SubElement(qff, "stepsize", value="0.5")
        ET.SubElement(qff, "ndifftype", value="hess")
        ET.SubElement(qff, "mopfile", value="prop_no_1.mop")

        tree = ET.ElementTree(root)
        ET.indent(tree, space="   ")
        tree.write(f"../output/{minfo}_output.xml", encoding="UTF-8", xml_declaration=True)

def create_run_script(minfo_files, output_file="../output/run.sh"):
    script_content = """#!/bin/bash

. /home/shun/SINDO/sindo/sindovars.sh
"""
    for minfo in minfo_files:
        xml_filename = f"{minfo}_output.xml"
        script_content += f"java RunMakePES -f {xml_filename} >& makePES.out\n"

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
