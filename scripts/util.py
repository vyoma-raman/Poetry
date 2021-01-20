import pickle5 as pickle
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from generators import existingGenerator

def savePoem(poem, name, accuracy, seed):
    filename = name + "-" + accuracy + "_" + "-".join(seed.split()) + ".txt"
    path = "../poems/" + filename
    with open(path, "w") as f:
        f.write(poem)
    print("The poem has been saved as '{}'.".format(filename))

def saveGenerator(gen):
    path = "../generators/" + gen.filename
    os.mkdir(path)
    # Save model
    gen.model.save(path + "/model")
    # Save tokenizer
    with open(path + "/tokenizer.pickle", "wb") as f:
        pickle.dump(gen.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Save max_sequence_len
    with open(path + "/max_len.txt", "w") as f:
        f.write(str(gen.max_sequence_len))

def loadGenerator(name, accuracy):
    filename = name + "-" + accuracy
    path = "../generators/" + filename
    # Load model
    model = tf.keras.models.load_model(path + "/model")
    # Load tokenizer
    with open(path + "/tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    # Load max_sequence_len
    with open(path + "/max_len.txt", "r") as f:
        max_sequence_len = int(f.read())
    return existingGenerator(filename, model, tokenizer, max_sequence_len, float(accuracy))
