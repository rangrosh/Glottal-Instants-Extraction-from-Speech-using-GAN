import numpy as np 
import matplotlib.pyplot as plt 
import librosa

import tensorflow as tf
from tensorflow.keras import models, layers 

import argparse 

import os
from tqdm import tqdm 

speech_path = 'Dataset/Speech'
egg_path = 'Dataset/EGG'
trim_size = 16384

def load_paths():

    speech = os.listdir(speech_path)
    egg = os.listdir(egg_path)
    names = [i for i in tqdm(speech, ascii = True, desc = "Loading Filepaths", ncols = 100) if i in egg]
    return names

def make_prediction(gen, inp, targ, prediction):

    out = gen(inp)

    plt.figure(figsize = (15, 20))

    plt.subplot(311)
    plt.plot(inp[0])
    plt.title("Input Signal")

    plt.subplot(312)
    plt.plot(targ[0])
    plt.title("Target Signal")

    plt.subplot(313)
    plt.plot(out[0])
    plt.title("Generated Signal")

    plt.suptitle("Prediction {}".format(prediction))
    # prediction += 1
    plt.show()

def main(modelPath = None, nPreds = None):
    if modelPath is None:
        raise OSError("Path to trained model not specified")
    if nPreds is None:
        nPreds = 1

    prediction = 1
    print(modelPath)
    model = models.load_model(modelPath)
    print(1)
    # gen_model, _ = model.layers 
    names = load_paths()
    for _ in range(nPreds):

        rand_idx = np.random.randint(0, len(names))
        input_signal = librosa.load(os.path.join(speech_path, names[_]))[0]
        target_signal = librosa.load(os.path.join(egg_path, names[_]))[0]
        l = len(input_signal)
        lim = l - trim_size
        low = np.random.randint(0, lim)
        high = low + trim_size 
        input_signal = input_signal[low:high]
        target_signal = target_signal[low:high]
        # print(_)
        input_signal = np.expand_dims(input_signal, -1)
        target_signal = np.expand_dims(target_signal, -1)
        make_prediction(model, np.expand_dims(input_signal, 0), np.expand_dims(target_signal, 0), prediction)
        prediction+=1


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--modelpath", required = True, help = "Path to Model")
    ap.add_argument("-n", "--predictions", required = True, help = "Number of Predictions to make")
    args = vars(ap.parse_args())
    path_to_model = str(args['modelpath'])
    num_preds = int(args['predictions'])
    main(modelPath = path_to_model, nPreds = num_preds)