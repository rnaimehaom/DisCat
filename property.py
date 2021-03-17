import time
import random

startTime = time.time()
with open("C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset\\final_dataset1.txt", "r") as doc:
    smiles = doc.readlines()
    
SMILES = []
for i in range(2000000):
    y = random.randint(0, 84645100)
    SMILES.append(smiles[y])
    if i % 100000 == 0:
        print(f'{i}, {SMILES[i]}')

print(len(smiles))
with open("ZINC_dataset\\final_dataset1.txt", 'w') as text:
    text.write("SMILES, Properties\n")
    for line in SMILES:
        text.write(line)
# with open('ZINC_dataset\\final_dataset1.txt', 'w') as test:
#     split = []
#     for i in range(len(smiles)):
#         if (i+1) % 3 != 0:
#             s = smiles[i].strip('\n')
#             split.append(s)
#         else:
#             s = smiles[i].strip('\n')
#             split.append(s)
#             line = ''.join(split)
#             test.write(f'{line}\n')
#             split.clear()
            

# with open("C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset\\properties.txt", "r") as prop:
#     properties = prop.readlines()

# for i, smi in enumerate(smiles):
#     smiles[i] = smi.strip("\n")
# for i in range(len(smiles)):
#     smiles[i] = f'{smiles[i]}, {properties[i]}\n'

# with open("C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset\\final_dataset1.txt", "w") as final:
#     for smi in smiles:
#         final.write(smi)
# smiles = []
# with open("C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset\\final_dataset1.txt", "r") as doc:
#     lines = doc.readlines()
# for line in lines:
#     split = line.split(" ")
#     zinc = split[1]
#     smi = split[0] + '\n'
#     smiles.append(smi)
#     with open("ZINC_id.txt", 'a') as z:
#         z.write(zinc)
# print("Done with ZINC!!!")
# ZINC_time = time.time() - startTime
# ZINC_time = ZINC_time/3600
# print(f"The ZINC took {ZINC_time} hours")

# with open('ZINC_dataset\\final_dataset1.txt', 'w') as data:
#     for smi in smiles:
#         data.write(smi)
# print("Done with SMILES!!!")
# whole_time = (time.time() - startTime) / 3600
# SMI_time = whole_time - ZINC_time
# print(f"SMILES took {SMI_time} hours")
# print(f"The whole script took {whole_time} hours")