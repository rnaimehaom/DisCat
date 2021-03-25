import yaml
import model_utilities as mu
import numpy as np

def vectorize_data(params):
    #define variables
    MAX_LEN = params['MAX_LEN']
    CHARS = yaml.safe_load(open(params['char_file']))
    params['NCHARS'] = len(CHARS)
    NCHARS = len(CHARS)
    CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))

    #Load data for properties
    if params['do_prop_pred'] and ('data_file' in params):
        if 'data_normalization_out' in params:
            normalize_out = params['data_normalization_out']
        else:
            normalize_out = None

        if "reg_prop_tasks" in params:
            smiles, Y_reg = mu.load_smiles_and_data_df(params['data_file'], MAX_LEN, 
                            reg_tasks=params['reg_prop_tasks'], normalize_out=normalize_out)
        
    else:
        smiles = mu.load_smiles_and_data_df(params['data_file'], MAX_LEN)
    
    if 'limit_data' in params.keys():
        sample_idx = np.random.choice(np.arange(len(smiles)), params['limit_data'], replace=False)
        smiles=list(np.array(smiles)[sample_idx])
        if params['do_prop_pred'] and ('data_file' in params):
            if 'reg_prop_tasks' in params:
                Y_reg = Y_reg[sample_idx]
    
    print(f"Training set size is {len(smiles)}")
    print(f'First smiles: \" {smiles[0]} \"')
    print(f'total chars: {NCHARS}')

    print("Vectorization...")

    X = mu.smiles_to_hot(smiles, MAX_LEN, params['PADDING'], CHAR_INDICES, NCHARS)

    print(f"Total Data size {X.shape[0]}")
    if np.shape(X)[0] % params['batch_size'] != 0:
        X = X[:np.shape(X)[0] // params['batch_size'] * params['batch_size']]
        if params['do_prop_pred']:
            if 'reg_prop_tasks' in params:
                Y_reg = Y_reg[:np.shape(Y_reg)[0] // params['batch_size'] * params['batch_size']]
    
    np.random.seed(params['RAND_SEED'])
    rand_idx = np.arange(np.shape(X)[0])
    np.random.shuffle(rand_idx)

    TRAIN_FRAC = 1 - params['val_split']
    num_train = int(X.shape[0] * TRAIN_FRAC)
    if num_train % params['batch_size'] != 0:
        num_train = (num_train // params['batch_size']) * params['batch_size']

    train_idx = rand_idx[:int(num_train)]
    test_idx = rand_idx[int(num_train):]

    if 'test_idx_file' in params.keys():
        np.save(params['test_idx_file'], test_idx)

    X_train = X[train_idx]
    X_test = X[test_idx]
    print(f'shape of input vector: {np.shape(X_train)}')
    print(f'Training set size is {np.shape(X_train)}, after filtering to max length of {MAX_LEN}')

    if params['do_prop_pred']:
        Y_train = []
        Y_test = []
        if 'reg_prop_tasks' in params:
            Y_reg_train = Y_reg[train_idx]
            Y_reg_test = Y_reg[test_idx]
            Y_train.append(Y_reg_train)
            Y_test.append(Y_reg_test)

        return X_train, X_test, Y_train, Y_test
    else:
        return X_train, X_test