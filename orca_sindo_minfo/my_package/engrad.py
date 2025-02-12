def engrad(dir="", name, atom_num):
    with open(f"{dir}{name}.engrad","r",encoding="UTF-8") as f:
        data = f.readlines()
    
    grad_index = next(i for i, line in enumerate(data) if "The current gradient" in line)
    grad_line = data[grad_index+2:grad_index+2+atom_num*3]
    grad_line = [line.strip() for line in grad_line]
    
    gradient = "\n".join(", ".join(grad_line[i:i+5]) for i in range(0, len(grad_line), 5))

    return gradient
