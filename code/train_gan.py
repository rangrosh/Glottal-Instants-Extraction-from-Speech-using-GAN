import numpy as np 
from matplotlib import pyplot as plt 
import random 
from scipy import signal 

import librosa 

import tensorflow as tf 
from tensorflow.keras import models, layers 
from tensorflow.keras import backend as K 

import os 
from tqdm import tqdm, trange
from statistics import mean

from segan_utils import Generator, Discriminator, Autoencoder

speech_path = 'Dataset/Speech'
egg_path = 'Dataset/EGG'

def load_reals():
    
    speech = os.listdir(speech_path)
    egg = os.listdir(egg_path)
    names = [i for i in tqdm(speech, ascii = True, desc = "Loading Filepaths", ncols = 110) if i in egg]
    random.shuffle(names)
    return names


def generate_outputs(names, batch_size, trim_size):

    idx = np.random.randint(0, len(names), batch_size).tolist()
    outputs = []
    for i in idx:
        x = librosa.load(os.path.join(egg_path, names[i]))[0]
        a = len(x)
        lim = a - trim_size 
        low = np.random.randint(0, lim)
        high = low + trim_size 
        x = x[low:high]
        outputs.append(np.expand_dims(x, -1))

    # outputs = [librosa.load(os.path.join(egg_path, names[i]))[0] for i in idx]
    return np.asarray(outputs)


def generate_inputs(names, batch_size, trim_size):
    #load inputs for generator
    #print(0)
    idx = np.random.randint(0, len(names), batch_size).tolist()
    #print(idx)
    #print(1)
    inputs = []
    for i in idx:
        x = librosa.load(os.path.join(speech_path, names[i]))[0]
        x = librosa.effects.preemphasis(x, coef = 0.95)
        a = len(x)
        #print(a)
        lim = a - trim_size 
        low = np.random.randint(0, lim)
        high = low + trim_size 
        x = x[low:high]
        inputs.append(np.expand_dims(x, -1))
    #print(3)
    # inputs = [librosa.load(os.path.join(speech_path, names[i]))[0] for i in idx]
    #print(inputs.shape)
    #print(inputs[0:10])
    xd = np.asarray(inputs)
    #print(xd.shape[0])
    #print(xd[3:7])
    # print(xd.shape)
    return xd


def generate_fakes(gen, names, batch_size, trim_size):
    #generate fake samples with generator
    gen_input = generate_inputs(names, batch_size, trim_size)
    #print(type(gen_input))
    #newar = np.asarray(gen_input).astype('float32')
    #y = gen.predict(gen_input)
    y = gen(gen_input)
    return y

def generate_input_output_pairs(names, batch_size, trim_size):
    
    idx = np.random.randint(0, len(names), batch_size).tolist()
    #print(idx)
    #print(1)
    inputs = []
    outputs = []
    for i in idx:
        x = librosa.load(os.path.join(speech_path, names[i]))[0]
        # x = librosa.effects.preemphasis(x, coef = 0.95)
        y = librosa.load(os.path.join(egg_path, names[i]))[0]
        a = len(x)
        #print(a)
        lim = a - trim_size 
        low = np.random.randint(0, lim)
        high = low + trim_size 
        x = x[low:high]
        y = y[low:high]
        inputs.append(np.expand_dims(x, -1))
        outputs.append(np.expand_dims(x, -1))
    #print(3)
    # inputs = [librosa.load(os.path.join(speech_path, names[i]))[0] for i in idx]
    #print(inputs.shape)
    #print(inputs[0:10])
    xd = np.asarray(inputs)
    yd = np.asarray(outputs)
    #print(xd.shape[0])
    #print(xd[3:7])
    # print(xd.shape)
    return xd, yd
class GAN:

    def __init__(self, input_shape = [1024, 1], batch_size = 40, trim_size = 1024, epochs = 100):

        self.input_shape = input_shape
        self.batch_size = batch_size
        self.trim_size = trim_size
        self.lr = 0.001
        self.gan = None
        self.gen = None 
        self.disc = None
        self.epochs = epochs
        self.lambda_ = 100 

    def genLSLoss(self, y_true, y_pred):

        loss_term = tf.keras.losses.mean_absolute_error(y_true, y_pred)
        # regularization_term = self.lambda_ * tf.norm(loss_term, ord = 1)
        return loss_term

    def discLSLoss(self, y_true, y_pred):

        return tf.keras.losses.mean_squared_error(y_true, y_pred)

    def get_gan(self):
        return self.gan
    
    def create_gan(self):

        self.gen = Generator().construct(input_shape = self.input_shape, batch_size = self.batch_size)
        self.gen.compile(loss = "mae", optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr))

        self.disc = Discriminator().construct(input_shape = self.input_shape)
        self.disc.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr))
        self.disc.trainable = False 

        self.gan = models.Sequential([self.gen, self.disc]) 

        self.gan.compile(loss = ["mae", "mse"], optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr))
 
    def make_prediction(self, gen, inp, actual):

        out = gen(inp)
        out = signal.lfilter([1], [1, -0.95], out).astype(np.float32)
        plt.figure(figsize = (15, 20))
        plt.subplot(311)
        plt.plot(inp[0])
        plt.title("Input Signal")
        plt.subplot(312)
        plt.plot(out[0])
        plt.title("Predicted EGG")
        plt.subplot(313)
        plt.plot(actual[0])
        plt.title("Actual EGG")
        plt.show()

    def save_model(self, gen, epoch):
        print("Checkpoint created at epoch {}".format(epoch + 1))
        gen.save('Checkpoint_{}_{}'.format(epoch + 1, self.epochs))

    def train_gan(self, checkpoint_freq = 5, plot_freq = 5):

        gen, disc = self.gan.layers 

        names = load_reals()[:100]
        n_batches = len(names)//self.batch_size

        for epoch in range(self.epochs):

            if(epoch % plot_freq == plot_freq - 1):
                random_idx = np.random.randint(0, len(names), self.batch_size)
                input_signal = []
                target_signal = []
                for __ in random_idx:
                        isig = librosa.load(os.path.join(speech_path, names[__]))[0]
                        isig = librosa.effects.preemphasis(isig, coef = 0.95)
                        tsig = librosa.load(os.path.join(egg_path, names[__]))[0]
                        l = len(isig)
                        lim = l - self.trim_size
                        low = np.random.randint(0, lim)
                        high = low + self.trim_size 
                        isig = isig[low:high]
                        tsig = tsig[low:high]
                        input_signal.append(np.expand_dims(isig, -1))
                        target_signal.append(np.expand_dims(tsig, -1))
                input_signal = np.asarray(input_signal)
                target_signal = np.asarray(target_signal)
                self.make_prediction(gen, input_signal, target_signal)


            training_loop = trange(n_batches, desc = 'Epoch {}/{}'.format(epoch + 1, self.epochs), ncols = 100)
            for i in training_loop:

                disc.trainable = True

                X_real = generate_outputs(names, self.batch_size, self.trim_size)
                y_real = np.ones((self.batch_size, 1))

                X_fake = generate_fakes(gen, names, self.batch_size, self.trim_size)
                y_fake = np.empty((self.batch_size, 1)) 
                y_fake.fill(-1)

                dr = disc.train_on_batch(X_real, y_real)
                df = disc.train_on_batch(X_fake, y_fake) 
                # disc_loss_real.append(dr)
                # disc_loss_fake.append(df)
                disc.trainable = False 

                X_auto = generate_input_output_pairs(names, self.batch_size, self.trim_size)

                al = gen.train_on_batch(X_auto[0], X_auto[1])

                X_gan = generate_inputs(names, self.batch_size, self.trim_size)
                y_gan = np.ones((self.batch_size, 1))

                gl = self.gan.train_on_batch(X_gan, y_gan)
                # gan_loss.append(gl)

                # mean_disc_loss = (mean(disc_loss_real) + mean(disc_loss_fake))/2
                # mean_gan_loss = mean(gan_loss)

                # training_loop.set_postfix({
                #     "Discriminator Loss" : (dr + df)/2, 
                #     "GAN Loss" : gl
                # })
                # disc_hist.append(mean_disc_loss)
                # gan_hist.append(mean_gan_loss)
                # gan_acc.append(ga)
            # print("Disc Loss: ", (mean(disc_loss_real) + mean(disc_loss_fake))/2)
            # print("GAN Loss: ", mean(gan_loss))
            # print("GAN Accuracy: ", gan_acc.mean())

            if((epoch+1) % checkpoint_freq == 0):
                self.save_model(gen, epoch)
            
class Autoenc:

    def __init__(self, input_shape = [1024, 1], batch_size = 32, trim_size = 1024, epochs = 150):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.trim_size = trim_size
        self.lr = 0.0002
        self.gan = None
        self.gen = None 
        self.disc = None
        self.epochs = epochs

    def create_autoencoder(self):

        self.autoencoder = Autoencoder().construct(input_shape = self.input_shape)
        self.autoencoder.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr), loss = tf.keras.losses.MeanSquaredError())

    def make_prediction(self, inp, actual):

        out = self.autoencoder.predict(inp)
        # print(inp.shape)
        # print(out.shape)
        # out = signal.lfilter([1], [1, -0.95], out).astype(np.float32)
        plt.figure(figsize = (15, 20))
        plt.subplot(311)
        plt.plot(inp[0])
        plt.title("Input Signal")
        plt.subplot(312)
        plt.plot(out[0])
        plt.title("Predicted EGG")
        plt.subplot(313)
        plt.plot(actual[0])
        plt.title("Actual EGG")
        plt.show()

    def load_data(self):
        dataset = 'Dataset'
        egg = []
        speech = []

        n = 0
        for i in tqdm(os.listdir(dataset + "/Speech")):
            n += 1
            egg_sample,_ = librosa.load(dataset + "/EGG" + "/" + i)
            speech_sample,_ = librosa.load(dataset + "/Speech" + "/" + i)
            egg.append(egg_sample)
            speech.append(speech_sample)
            # if n == 100:
            #     break
        return speech, egg

    def trim_data(self, speech, egg):
        egg_dataset = []
        speech_dataset = []

        for i in range(len(egg)):
            for j in range(len(egg[i])//self.trim_size):
                egg_dataset.append(egg[i][self.trim_size*j:self.trim_size*(j+1)])
                speech_dataset.append(speech[i][self.trim_size*j:self.trim_size*(j+1)])

        egg_dataset = np.array(egg_dataset)
        speech_dataset = np.array(speech_dataset)
        return speech_dataset, egg_dataset

    def save_model(self, epoch):
        print("Checkpoint created at epoch {}".format(epoch))
        self.autoencoder.save('Checkpoint_{}_{}'.format(epoch, self.epochs))

    def train_autoencoder(self, checkpoint_freq = 5, plot_freq = 5):

        names = load_reals()

        speech_dataset, egg_dataset = self.load_data()
        speech_dataset, egg_dataset = self.trim_data(speech_dataset, egg_dataset)
        n = len(speech_dataset)
        X_train = speech_dataset[:int(n*0.95)]
        Y_train = egg_dataset[:int(n*0.95)]
        X_val = speech_dataset[int(n*0.95):int(n*0.975)]
        Y_val = egg_dataset[int(n*0.95):int(n*0.975)]
        X_test = speech_dataset[int(n*0.975):]
        Y_test = egg_dataset[int(n*0.975):]

        del egg_dataset,speech_dataset

        n_batches = len(names)//self.batch_size

        # X_auto = generate_input_output_pairs(names, len(names), self.trim_size)
        n_epochs_per_plot = self.epochs//plot_freq 
        # for epoch in range(n_epochs_per_plot)
        for i in range(self.epochs//checkpoint_freq):
            self.autoencoder.fit(
                X_train, Y_train, 
                batch_size = self.batch_size, 
                epochs = checkpoint_freq, 
                shuffle = True,
                validation_data = (X_val, Y_val)
            )

            random_idx = np.random.randint(0, len(names))
            input_signal = []
            target_signal = []
            input_signal = librosa.load(os.path.join(speech_path, names[random_idx]))[0]
            target_signal = librosa.load(os.path.join(egg_path, names[random_idx]))[0]
            l = len(input_signal)
            lim = l - self.trim_size
            low = np.random.randint(0, lim)
            high = low + self.trim_size 
            input_signal = input_signal[low:high]
            target_signal = target_signal[low:high]
            input_signal = np.asarray(input_signal)
            target_signal = np.asarray(target_signal)
            input_signal = np.expand_dims(input_signal, -1)
            target_signal = np.expand_dims(target_signal, -1)
            self.make_prediction(np.expand_dims(input_signal, 0), np.expand_dims(target_signal, 0))
            self.save_model((i+1)*checkpoint_freq)
        # for epoch in range(self.epochs):

        #     if(epoch % plot_freq == plot_freq - 1):
        #         random_idx = np.random.randint(0, len(names), self.batch_size)
        #         input_signal = []
        #         target_signal = []
        #         for __ in random_idx:
        #                 isig = librosa.load(os.path.join(speech_path, names[__]))[0]
        #                 # isig = librosa.effects.preemphasis(isig, coef = 0.95)
        #                 tsig = librosa.load(os.path.join(egg_path, names[__]))[0]
        #                 l = len(isig)
        #                 lim = l - self.trim_size
        #                 low = np.random.randint(0, lim)
        #                 high = low + self.trim_size 
        #                 isig = isig[low:high]
        #                 tsig = tsig[low:high]
        #                 input_signal.append(np.expand_dims(isig, -1))
        #                 target_signal.append(np.expand_dims(tsig, -1))
        #         input_signal = np.asarray(input_signal)
        #         target_signal = np.asarray(target_signal)
        #         self.make_prediction(input_signal, target_signal)

        #     # training_loop = trange(n_batches, desc = 'Epoch {}/{}'.format(epoch + 1, self.epochs), ncols = 100)
        #     # for i in training_loop:
        #     for i in range(n_batches):
        #         X_auto = generate_input_output_pairs(names, self.batch_size, self.trim_size)
        #         # al = self.autoencoder.train_on_batch(X_auto[0], X_auto[1])
        #         # training_loop.set_postfix({
        #         #     "Loss" : al
        #         # })
        #         self.autoencoder.fit(X_auto[0], X_auto[1], epochs = 1)
        #     if((epoch+1) % checkpoint_freq == 0):
        #         self.save_model(epoch)

