import keras.layers as KL
import keras.initializers as KI
import keras.models as KM
import keras.optimizers as KO
import math
import numpy as np
import os
import cv2


BATCH_SIZE = 64


def LeakyRelu(x):
    return KL.LeakyReLU(alpha=0.2)(x)

def BN(x):
    return KL.BatchNormalization(momentum=0.99, epsilon=1e-5,
                                 gamma_initializer=KI.random_normal(mean=1., stddev=0.02))(x)

def Conv(x, f_dim):
    return KL.Conv2D(filters=f_dim,
                     kernel_size=(5, 5),
                     strides=(2, 2),
                     padding='same',
                     kernel_initializer=KI.truncated_normal(stddev=0.02))(x)

def Linear(x, output_size, activate_mode=None):
    return KL.Dense(output_size,
                    kernel_initializer=KI.random_normal(stddev=0.02),
                    activation=activate_mode)(x)

def Deconv(x, f_dim):
    return KL.Deconv2D(filters=f_dim,
                       kernel_size=(5, 5),
                       strides=(2, 2),
                       kernel_initializer=KI.random_normal(stddev=0.02),
                       padding='same')(x)


class DCGAN():
    def __init__(self, dataset, input_size=96, df_dim=64, gf_dim=64, c_dim=3):
        self.dataset = dataset
        self.input_size = input_size
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.c_dim = c_dim
        self.d_model, self.g_model, self.d_on_g_model = self.build()

    def train(self):
        x_train = self.dataset
        x_train = (x_train.astype(np.float32) - 127.5)/127.5
        d_optim = KO.Adam(lr=0.0002, beta_1=0.5)
        d_on_g_optim = KO.Adam(lr=0.0002, beta_1=0.5)
        self.g_model.compile(loss='binary_crossentropy', optimizer='SGD')
        self.d_on_g_model.compile(loss='binary_crossentropy', optimizer=d_on_g_optim)
        self.d_model.trainable = True
        self.d_model.compile(loss='binary_crossentropy', optimizer=d_optim)

        for epoch in range(100):
            print("Epoch is", epoch)
            print("Number of batches", int(x_train.shape[0] / BATCH_SIZE))
            for index in range(int(x_train.shape[0] / BATCH_SIZE)):
                noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
                image_batch = x_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                generated_images = self.g_model.predict(noise, verbose=0)
                if index % 700 == 0:
                    image = self.combine_images(generated_images)
                    image = image * 127.5 + 127.5
                    cv2.imwrite(str(epoch) + "_" + str(index) + ".png", image)
                X = np.concatenate((image_batch, generated_images))
                y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
                self.d_model.trainable = True
                d_loss = self.d_model.train_on_batch(X, y)
                print("batch %d d_loss : %f" % (index, d_loss))
                self.d_model.trainable = False
                for _ in range(2):
                    noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
                    g_loss = self.d_on_g_model.train_on_batch(noise, [1] * BATCH_SIZE)
                    print("batch %d g_loss : %f" % (index, g_loss))
                if index % 200 == 199:
                    self.g_model.save_weights('generator', True)
                    self.d_model.save_weights('discriminator', True)

    def discriminator(self, input_tensor):
        f_dim = self.df_dim
        h0 = LeakyRelu(Conv(input_tensor, f_dim))
        h1 = LeakyRelu(BN(Conv(h0, f_dim*2)))
        h2 = LeakyRelu(BN(Conv(h1, f_dim*4)))
        h3 = LeakyRelu(BN(Conv(h2, f_dim*8)))
        h4 = Linear(KL.Flatten()(h3), output_size=1, activate_mode='sigmoid')
        return h4

    def generator(self, input_tensor):
        h_dim = int(self.input_size / 16)
        f_dim = self.gf_dim
        x = Linear(input_tensor, f_dim*8*h_dim*h_dim)
        x = KL.Reshape([h_dim, h_dim, f_dim*8])(x)
        h0 = KL.Activation('relu')(BN(x))
        h1 = KL.Activation('relu')(BN(Deconv(h0, f_dim*4)))
        h2 = KL.Activation('relu')(BN(Deconv(h1, f_dim*2)))
        h3 = KL.Activation('relu')(BN(Deconv(h2, f_dim)))
        h4 = KL.Activation('tanh')(Deconv(h3, self.c_dim))
        return h4

    def build(self):
        # build d graph
        image_tensor = KL.Input(shape=(self.input_size, self.input_size, self.c_dim),
                                name='real_image')
        d_out = self.discriminator(image_tensor)
        d_model = KM.Model(image_tensor, d_out, name='D')

        # build g graph
        z_tensor = KL.Input(shape=(100,), name='random_noise')
        g_out = self.generator(z_tensor)
        g_model = KM.Model(z_tensor, g_out, name='G')

        # build d_on_g graph
        model = KM.Sequential()
        model.add(g_model)
        d_model.trainable = False
        model.add(d_model)
        d_on_g_model = model
        return d_model, g_model, d_on_g_model

    def combine_images(self, generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
        shape = generated_images.shape[1:3]
        image = np.zeros((height * shape[0], width * shape[1], self.c_dim),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index / width)
            j = index % width
            image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
                img[:, :, :]
        return image


def load_data(filepath=None):
    if filepath == None:
        datapth = os.path.join(os.getcwd(), 'data/faces')
        filename = os.listdir(datapth)
        num = len(filename)
        dataset = np.zeros([num, 96, 96, 3])
        for i in range(num):
            dataset[i, :, :, :] = cv2.imread(os.path.join(datapth, filename[i]))
        np.save('faces.npy', dataset)
    else:
        dataset = np.load(filepath)
    return dataset


if __name__ == '__main__':
    dataset = load_data('faces.npy')
    dcgan = DCGAN(dataset)
    dcgan.train()