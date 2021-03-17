import json
from collections import OrderedDict

def load_params(param_file=None, verbose=True):
    # Parameters from params.json and exp.json loaded here to override parameters set below
    if param_file is not None:
        hyper_p = json.loads(open(param_file).read(),
                             object_pairs_hook=OrderedDict)
        if verbose:
            print('Using hyper-parameters:')
            for key, value in hyper_p.items():
                print('{:25s} - {:12}'.format(key, str(value)))
            print('rest of parameters are set as default')
    parameters = {

        'reload_model' : False,
        'prev_epochs' : 0,

    'batch_size' : 100,
    'epochs' : 1,
    "val_split" : 0.2,
    'loss' : 'categorical_crossentropy',

    'batchnorm_conv' : True,
    'conv_activation' : 'tanh',
    'conv_depth' : 4,
    'conv_dim_width' : 8,
    'conv_dim_depth' : 8,
    'conv_d_growth_factor' : 1.15875438383,
    'conv_w_growth_factor' : 1.1758149644,

    'gru_depth' : 4,
    'rnn_activation' : 'tanh',
    'recurrent_dim' : 50,
    'do_tgru' : True,
    'terminal_GRU_implementation' : 0,
    'tgru_dropout' : 0.0,
    'temperature' : 1.00,

    'hg_growth_factor' : 1.4928245388,
    'hidden_dim' : 100,
    'middle_layer' : 1,
    'dropout_rate_mid' : 0.0,
    'batchnorm_mid' : True,
    'activation' : 'tanh',

    'lr' : 0.000312087049936,
    'momentum' : 0.936948773087,
    'optim' : 'adam',

    'vae_annealer_start' : 22,
    'batchnorm_vae' : False,
    'vae_activation' : 'tanh',
    'xent_loss_weight' : 1.0,
    'kl_loss_weight' : 1.0,
    'anneal_sigmoid_slope' : 1.0,
    'freeze_logvar_layer' : False,
    'freeze_offset' : 1,

    'do_prop_pred' : False,
    'prop_pred_depth' : 3,
    'prop_hidden_dim' : 36,
    'prop_growth_factor' : 0.8,
    'prop_pred_activation' : 'tanh',
    'reg_prop_pred_loss' : 'mse',
    'logit_prop_pred_loss' : 'binary_crossentropy',
    'prop_pred_loss_weight' : 0.5,
    'prop_pred_dropout' : 0.0,
    'prop_batchnorm' : True,

    'verbose_print' : 0,
    }
    parameters.update(hyper_p)
    return parameters