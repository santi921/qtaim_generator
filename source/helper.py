# open file fail_id.txt
import os 

root = "../data/hydro/QTAIM/"
# iterate through each line in the file
with open("./fail_id.txt") as file: 
    lines = file.readlines()

for line in lines: 
    print(line)
    try:
        # remove file in root + id
        os.remove(root + line.strip() + "/reactants/" + "CPprop.txt")
        os.remove(root + line.strip() + "/products/" + "CPprop.txt")
        print("deleted!")
    except: print("failed")