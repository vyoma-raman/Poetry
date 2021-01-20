import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

class Callback(tf.keras.callbacks.Callback):
    """Callback object that stops training when minimum accuracy reached."""
    def __init__(self, accuracy):
        self.accuracy = float(accuracy)

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > self.accuracy:
            print("\nStopping training after reaching %2.2f%% accuracy." %(self.accuracy * 100))
            self.model.stop_training = True