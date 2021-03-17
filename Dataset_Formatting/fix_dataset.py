import os

for root, dirs, files in os.walk("C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset"):
    for file in files:
        with open(f"C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset\\{file}", 'r') as doc:
            lines = doc.readlines()
            lines[-1] = lines[-1].strip("\n")
        with open(f"C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset\\{file}", 'w') as write:
            for line in lines:
                write.write(line)