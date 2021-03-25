from vae_utils import VAEUtils
import model_utilities as mu
import pandas as pd
import random

def main(BE_Value):
    vae = VAEUtils(directory="C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\zinc_properties")

    df = pd.read_csv("C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset\\final_dataset2.csv")
    SMILES = df.iloc[:, 0]
    prop_val = BE_Value

    molecules, prop_num, values= find_molecule(prop_val, SMILES, vae)
    test = True
    while test:
        if abs(prop_val - prop_num) > 0.5:
            molecules, prop_num, values = find_molecule(prop_val, SMILES, vae)
        else:
            test = False

    for i in range(len(values)):
        if prop_num == values[i]:
            final_molecule = molecules[i]
            break
    return final_molecule

def closest(lst, K): 
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def find_molecule(prop_val, SMILES, vae):
    molecules = []
    for i in range(200):
        j = random.randint(0, len(SMILES))
        molecules.append(SMILES[j])

    values = []
    for i in range (len(molecules)):

        smiles_1 = mu.canon_smiles(molecules[i])

        X_1 = vae.smiles_to_hot(smiles_1,canonize_smiles=True)
        z_1 = vae.encode(X_1)
        X_r = vae.decode(z_1)
        molecules[i] = vae.hot_to_smiles(X_r,strip=True)[0]

        y_1 = vae.predict_prop_Z(z_1)[0]
        values.append(y_1)
    
    prop_num = closest(values, prop_val)
    return molecules, prop_num, values