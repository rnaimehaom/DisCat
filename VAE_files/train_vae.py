import argparse
import numpy as np
import tensorflow as tf 
import time
from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
import parameters
import model_callbacks as mol_cb 
from keras.callbacks import CSVLogger
from VAE_models import encoder_model, load_encoder, decoder_model, load_decoder
from VAE_models import property_predictor_model, load_property_predictor
from VAE_models import variational_layers
from preprocess import vectorize_data
from functools import partial
from keras.layers import Lambda

def load_models(params):
    def identity(x):
        return K.identity(x)
    
    kl_loss_var = K.variable(params['kl_loss_weight'], constraint=None)

    if params['reload_model'] == True:
        encoder = load_encoder(params)
        decoder = load_decoder(params)
    else:
        encoder = encoder_model(params)
        decoder = decoder_model(params)
    
    x_in = encoder.inputs[0]

    z_mean, enc_output = encoder(x_in)
    z_samp, z_mean_log_var_output = variational_layers(z_mean, enc_output, kl_loss_var, params)

    #Decoder
    x_out = decoder([z_samp,x_in])

    x_out = Lambda(identity, name='x_pred')(x_out)
    model_outputs = [x_out, z_mean_log_var_output]

    AE_only_model = Model(x_in, model_outputs)

    if params['do_prop_pred']:
        if params['reload_model'] == True:
            property_predictor = load_property_predictor(params)
        else:
            property_predictor = property_predictor_model(params)
        
        if ('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0):
            reg_prop_pred = property_predictor(z_mean)
            reg_prop_pred = Lambda(identity, name='reg_prop_pred')(reg_prop_pred)
            model_outputs.append(reg_prop_pred)
        else:
            raise ValueError('No regression tasks specified for property prediction')

        AE_PP_model = Model(x_in, model_outputs)
        return AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var

    else:
        return AE_only_model, encoder, decoder, kl_loss_var
    
def kl_loss(truth_dummy, x_mean_log_var_output):
    x_mean, x_log_var = tf.split(x_mean_log_var_output, 2, axis=1)
    print(f'x_mean shape is kl_loss: {x_mean.get_shape()}')
    kl_loss = -0.5 * K.mean(1 + x_log_var - K.square(x_mean) - 
            K.exp(x_log_var), axis=1)
    return kl_loss

def main_no_prop(params):
    start_time = time.time()

    X_train, X_test = vectorize_data(params)
    AE_only_model, encoder, decoder, kl_loss_var = load_models(params)

    # compile models
    if params['optim'] == 'adam':
        optim = Adam(lr=params['lr'], beta_1=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optim = RMSprop(lr=params['lr'], rho=params['momentum'])
    elif params['optim'] == 'sgd':
        optim = SGD(lr=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplementedError("Please define valid optimizer")

    model_losses = {'x_pred': params['loss'],
                        'z_mean_log_var': kl_loss}

    # vae metrics, callbacks
    vae_sig_schedule = partial(mol_cb.sigmoid_schedule, slope=params['anneal_sigmod_slope'],
                               start=params['vae_annealer_start'])
    vae_anneal_callback = mol_cb.WeightAnnealer_epoch(
            vae_sig_schedule, kl_loss_var, params['kl_loss_weight'], 'vae' )

    csv_clb = CSVLogger(params["history_file"], append=False)
    callbacks = [ vae_anneal_callback, csv_clb]


    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var

    xent_loss_weight = K.variable(params['xent_loss_weight'])
    model_train_targets = {'x_pred':X_train,
                'z_mean_log_var':np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))}
    model_test_targets = {'x_pred':X_test,
        'z_mean_log_var':np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))}

    AE_only_model.compile(loss=model_losses,
        loss_weights=[xent_loss_weight,
          kl_loss_var],
        optimizer=optim,
        metrics={'x_pred': ['categorical_accuracy',vae_anneal_metric]}
        )

    keras_verbose = params['verbose_print']

    AE_only_model.fit(X_train, model_train_targets,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    initial_epoch=params['prev_epochs'],
                    callbacks=callbacks,
                    verbose=keras_verbose,
                    validation_data=[ X_test, model_test_targets]
                    )

    encoder.save(params['encoder_weights_file'])
    decoder.save(params['decoder_weights_file'])
    print('time of run : ', time.time() - start_time)
    print('**FINISHED**')
    return

def main_property_run(params):
    start_time = time.time()

    #load_data
    X_train, X_test, Y_train, Y_test = vectorize_data(params)

    #load full models:
    AE_only_model, AE_PP_model, encoder, decoder, property_predictor, kl_loss_var = load_models(params)

    #compile models:
    if params['optim'] == 'adam':
        optim = Adam(lr=params['lr'], beta_1=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optim = RMSprop(lr=params['lr'], rho=params['momentum'])
    elif params['optim'] =='sgd':
        optim = SGD(lr=params['lr'], momentum=params['momentum'])
    else:
        raise NotImplementedError("Please define a valid optimizer")
    model_train_targets = {'x_pred':X_train, 
            'z_mean_log_var':np.ones((np.shape(X_train)[0], params['hidden_dim'] * 2))}
    model_test_targets = {'x_pred':X_test,
            'z_mean_log_var':np.ones((np.shape(X_test)[0], params['hidden_dim'] * 2))}
    model_losses = {'x_pred': params['loss'], 'z_mean_log_var': kl_loss}

    xent_loss_weight = K.variable(params['xent_loss_weight'])
    ae_loss_weight = 1. - params['prop_pred_loss_weight']
    model_loss_weights = {'x_pred': ae_loss_weight*xent_loss_weight,
                        'z_mean_log_var': ae_loss_weight*kl_loss_var}
    prop_pred_loss_weight = params['prop_pred_loss_weight']

    print(np.shape(Y_test[0]))
    if('reg_prop_tasks' in params) and (len(params['reg_prop_tasks']) > 0):
        model_train_targets['reg_prop_pred'] = Y_train[0]
        model_test_targets['reg_prop_pred'] = Y_test[0]
        model_losses['reg_prop_pred'] = params['reg_prop_pred_loss']
        model_loss_weights['reg_prop_pred'] = prop_pred_loss_weight
    
    #vae metrics, callbacks
    vae_sig_schedule = partial(mol_cb.sigmoid_schedule, slope=params['anneal_sigmoid_slope'],
                            start=params['vae_annealer_start'])
    vae_anneal_callback = mol_cb.WeightAnnealer_epoch(vae_sig_schedule, kl_loss_var, params['kl_loss_weight'], 'vae')

    csv_clb = CSVLogger(params['history_file'], append=False)

    callbacks = [vae_anneal_callback, csv_clb]
    def vae_anneal_metric(y_true, y_pred):
        return kl_loss_var
    
    keras_verbose = params['verbose_print']

    if 'checkpoint_path' in params.keys():
        callbacks.append(mol_cb.EncoderDecoderCheckpoint(encoder, decoder,
                        params=params, prop_pred_model=property_predictor, save_best_only=True))
    
    AE_PP_model.compile(loss=model_losses, loss_weights=model_loss_weights,
                        optimizer=optim,
                        metrics={'x_pred': ['categorical_accuracy', vae_anneal_metric]})

    print(f"Shape of reg_prop_pred in model_test_targets is {np.shape(model_train_targets['reg_prop_pred'])}")
    print(f"Shape of X_test in model_test_targets is {np.shape(model_test_targets['x_pred'])}")
    print(f"Shape of z_mean_log_var in model_test_targets is {np.shape(model_test_targets['z_mean_log_var'])}")
    
    AE_PP_model.fit(X_train, model_train_targets, 
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    initial_epoch=params['prev_epochs'],
                    callbacks=callbacks,
                    verbose=keras_verbose,
                    validation_data=[X_test, model_test_targets])
    
    encoder.save(params['encoder_weights_file'])
    decoder.save(params['decoder_weights_file'])
    property_predictor.save(params['prop_pred_weights_file'])

    print(f'time of run: {time.time() - start_time}')
    print("Finally................ ***FINISHED***")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_file',
                        help='experiment file', default='exp.json')
    args = vars(parser.parse_args())
    params = parameters.load_params(args['exp_file'])

    if params['do_prop_pred']:
        main_property_run(params)
    else:
        main_no_prop(params)