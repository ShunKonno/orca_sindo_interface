def energy(dir="", name):Total Energy       :
    with open(f"{mol}_opt.out","r",encoding="UTF-8") as f:
        data = f.readlines()
    
    energy_index = next(i for i, line in enumerate(data) if "Total Energy       :" in line)
    energy = float(data[energy_index].split(":")[1].split("Eh")[0])
    
    return energy

def charge
