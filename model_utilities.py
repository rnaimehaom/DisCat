import pandas as pd
import numpy as np
import logging
from rdkit.Chem import AllChem as Chem

def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        pass
    return None

def verify_smiles(smile):
    return (smile != '') and pd.notnull(smile) and (Chem.MolFromSmiles(smile) is not None)

def good_smiles(smile):
    if verify_smiles(smile):
        return canon_smiles(smile)
    else:
        return None

def pad_smile(string, max_len, padding='right'):
    if len(string) <= max_len:
        if padding == 'right':
            return string + " " * (max_len - len(string))
        elif padding == 'left':
            return " " * (max_len - len(string)) + string
        elif padding == 'none':
            return string

def filter_valid_length(strings, max_len):
    return [s for s in strings if len(s) <= max_len]

def filter_valid_smiles_return_invalid(strings, max_len):
    filter_list = []
    new_smiles = []
    for idx, s in enumerate(strings):
        if len(s) > max_len:
            filter_list.append(idx)
        else:
            new_smiles.append(s)
        return new_smiles, filter_list

def smiles_to_hot(smiles, max_len, padding, char_indices, nchars):
    smiles = [pad_smile(i, max_len, padding) for i in smiles if pad_smile(i, max_len, padding)]
    X = np.zeros((len(smiles), max_len, nchars), dtype=np.float32)
    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            try:
                X[i, t, char_indices[char]] = 1
            except KeyError as e:
                print(f"ERROR: Check chars file. Bad SMILES: {smile}")
                raise e
    return X

def smiles_to_hot_filter(smiles, char_indices):
    filtered_smiles = []
    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            try:
                char_indices[char]
            except KeyError:
                break
        else:
            filtered_smiles.append(smile)
    return filtered_smiles

def term_hot_to_smiles(x, temperature, indices_chars):
    temp_string = ""
    for j in x:
        index = thermal_argmax(j, temperature)
        temp_string += indices_chars[index]
    return temp_string

def hot_to_smiles(hot_x, indices_chars):
    smiles = []
    for x in hot_x:
        temp_str = ""
        for j in x:
            index = np.argmax(j)
            temp_str += indices_chars[index]
        smiles.append(temp_str)
    return smiles

def thermal_argmax(prob_arr, temperature):
    prob_arr = np.log(prob_arr) / temperature
    prob_arr = np.exp(prob_arr) / np.sum(np.exp(prob_arr))
    print(prob_arr)
    if np.greater_equal(prob_arr.sum(), 1.0000000001):
        logging.warn('Probabilities to sample add to more than 1, {}'.
                     format(prob_arr.sum()))
        prob_arr = prob_arr / (prob_arr.sum() + .0000000001)
    if np.greater_equal(prob_arr.sum(), 1.0000000001):
        logging.warn('Probabilities to sample still add to more than 1')
    return np.argmax(np.random.multinomial(1, prob_arr, 1))

def load_smiles(smi_file, max_len=None, return_filtered=False):
    if smi_file == "C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset\\final_dataset1.txt":
        with open("C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\ZINC_dataset\\final_dataset1.txt", 'r') as f:
            smiles = f.readlines()
        smiles = [i.strip() for i in smiles]

    if max_len is not None:
        if return_filtered:
            smiles, filtrate = filter_valid_smiles_return_invalid(smiles, max_len)
            if len(filtrate) > 0:
                print(f"Filtered {len(filtrate)} smiles due to length")
            return smiles, filtrate
        else:
            old_len = len(smiles)
            smiles = filter_valid_length(smiles, max_len)
            diff_len = old_len - len(smiles)
            if diff_len != 0:
                print(f"Filtered {diff_len} smiles due to length")
    return smiles

def convert_dtype(x):
    return float(x)

def load_smiles_and_data_df(data_file, max_len, reg_tasks=None, normalize_out=None):
    df = pd.read_csv(data_file, low_memory=False)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip()
    df = df[df.iloc[:, 0].str.len() <= max_len]
    smiles = df.iloc[:, 0].tolist()
    
    BE_dataframe = df['Properties']

    if len(reg_tasks) != 0:
        return smiles, np.vstack(BE_dataframe.values)
    else:
        return smiles

def smiles2one_hot_chars(smi_list, max_len):
    char_lists = [list(smi) for smi in smi_list]
    chars = list(set([char for sub_list in char_lists for char in sub_list]))
    chars.append(' ')
    return chars

def make_charset(smi_file, char_file):
    with open(smi_file, 'r') as afile:
        unique_chars = set(afile.read())
    bad = ['\n', '"']
    unique_chars = [c for c in unique_chars if c not in bad]
    unique_chars.append(' ')
    print(f"Found {len(unique_chars)} unique chars")
    astr = str(unique_chars).replace("\'", "\"")
    print(astr)

    with open(char_file, 'w') as bfile:
        bfile.write(astr)
    return

def CheckSmiFeasible(smi):
    try:
        get_molecule_smi(Chem.MolFromSmiles(smi))
    except:
        return False
    return True

def balanced_parentheses(input_string):
    s = []
    balanced = True
    index = 0
    while index < len(input_string) and balanced:
        token = input_string[index]
        if token == "(":
            s.append(token)
        elif token ==")":
            if len(s) == 0:
                balanced = False
            else:
                s.pop()

        index +=1
    return balanced and len(s) == 0

def matched_ring(s):
    return s.count('1') % 2 == 0 and s.count('2') % 2 == 0

def fast_verify(s):
    return matched_ring(s) and balanced_parentheses(s)

def get_molecule_smi(mol_obj):
    return Chem.MolToSmiles(mol_obj)

def canon_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)

if __name__ == '__main__':
    smiles, bindingEnergies = load_smiles_and_data_df("ZINC_dataset\\final_dataset2.csv", 120, ['Binding_energies'])
    print(smiles[:5])
    print(bindingEnergies[:5])