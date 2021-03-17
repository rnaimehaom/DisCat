from keras.callbacks import Callback, ModelCheckpoint
import numpy as np
from keras import backend as K

class WeightAnnealer_epoch(Callback):
    def __init__(self, schedule, weight, weight_orig, weight_name):
        super(WeightAnnealer_epoch, self).__init__()
        self.schedule = schedule
        self.weight_var = weight
        self.weight_orig = weight_orig
        self.weight_name = weight_name

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        new_weight = self.schedule(epoch)
        new_value = new_weight * self.weight_orig
        print(f"Current {self.weight_name} anealer weight is {new_value}")
        assert type(
           new_weight) == float, 'The output of the "schedule" function should be float.'
        K.set_value(self.weight_var, new_value)

def no_schedule(epoch_num):
    return float(1)

def sigmoid_schedule(time_step, slope=1, start=None):
    return float(1 / (1. + np.exp(slope * (start - float(time_step)))))

def sample(a, temperature=0.01):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.nultinomial(1, a, 1))

class EncoderDecoderCheckpoint(ModelCheckpoint):
    def __init__(self, encoder_model, decoder_model, 
                params, prop_pred_model=None, prop_to_monitor='val_x_pred_categorical_accuracy', 
                save_best_only=True, monitor_op=np.greater, monitor_best_init=-np.Inf):
        self.p = params
        super().__init__('weights.hdf5')
        self.save_best_only = save_best_only
        self.monitor = prop_to_monitor
        self.monitor_op = monitor_op
        self.best = monitor_best_init
        self.verbose = 1
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.prop_pred_model = prop_pred_model

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
    
        if self.save_best_only:
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(f"Epoch {epoch}: {self.monitor} improved from {self.best} to {current}\n Saving model")
                self.best = current
                self.encoder.save(f"C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\encoder_{epoch}.h5") 
                self.decoder.save(f"C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\decoder_{epoch}.h5")
                if self.prop_pred_model is not None:
                    self.prop_pred_model.save(f"C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\prop_pred_model_{epoch}.h5")
            else:
                if self.verbose > 0:
                    print(f"Epoch {epoch}: {self.monitor} did not improve")
        else:
            if self.verbose > 0:
                print(f"Epoch {epoch}: saving model")
            self.encoder.save(f"C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\encoder_{epoch}.h5") 
            self.decoder.save(f"C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\decoder_{epoch}.h5")
            if self.prop_pred_model is not None:
                self.prop_pred_model.save(f"C:\\Users\\Roshan\\Documents\\Science_Fair_2020-2021\\prop_pred_model_{epoch}.h5")