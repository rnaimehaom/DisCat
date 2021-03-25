import os
# import xlrd
import pandas as pd

df = pd.read_csv('history.csv')
print(df.head())
# f = pd.read_csv('ZINC_dataset\\final_dataset1.txt')
# f.to_csv('ZINC_dataset\\final_dataset2.csv', index=None)

# with open("C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset\\final_dataset1.txt", 'r') as doc:
#     hello = doc.readlines()
# for i in range(len(hello)):
#     words = hello[i].split(",")
#     SMILES = words[0]
#     propert = words[1]
#     if propert == ' -0.0\n':
#         if SMILES == 'C=CCNC(=O)[C@@H](C#N)C(=O)c1cccc(CN(C)C)c1':
#             print(propert)
#         hello[i] = f"{SMILES}, -0.1\n"
    
# with open("C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset\\final_dataset1.txt", 'w') as f:
#     for line in hello:
#         f.write(line)
# for root, dirs, files in os.walk("C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset"):
#     for file in files:
#         if file == "final_dataset.txt" or "final_dataset1.txt":
#             continue
#         new_file = open("C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset\\final_dataset1.txt", "a")
#         with open(f"C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset\\{file}", 'r') as doc:
#             new_file.write(doc.read())
#         new_file.close()
#         doc.close()