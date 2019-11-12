# -*- coding: utf-8 -*-
#!/usr/bin/env python3


import tensorflow as tf
import numpy as np
import time
import scipy.misc
import imageio

from utils.pcolor import PColor

from utils import image_loader #import choose_random_batch
choose_random_batch = image_loader.choose_random_batch
from geo_maker import geometry_maker #import simple_triangles
simple_triangles = geometry_maker.simple_triangles

RGB_CHANNELS = 3
class SessionSaver:
    def __init__(self, sess, RESET_FRESH):
        # self.session_saver_init(sess, RESET_FRESH)

        #def session_saver_init(self, sess, RESET_FRESH):

        self.SAVED_SESSION_FILENAME = "./trained_session.ckpt"

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        if RESET_FRESH:
            print("resetting training. Not loading previous training session.")
        else:
            print("Loading previous training session.")

        if not RESET_FRESH:
            sself.saver.restore(sess, self.SAVED_SESSION_FILENAME)

    def tick(self, sess):
        self.save_path = self.saver.save(sess, self.SAVED_SESSION_FILENAME)

"""
from mnist_works import loadmnist_from_args
img, label, RESET_FRESH = loadmnist_from_args()
"""



# IMPORTANT DESIGN CHOICE
# 7, 14, 20, 28, 200
RGB_SIZE = (14,14, RGB_CHANNELS)

FLATTENED_SIZE = np.prod(np.array(RGB_SIZE))


# img, label, RESET_FRESH = simple_traiangles()
main_artworks = simple_triangles(FLATTENED_SIZE/RGB_CHANNELS, RGB_CHANNELS)


PColor.init()

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
#  IMPORTANT DESIGN CHOICES

#LR_G = 0.0001          # learning rate for generator
LR_G = 0.001            # learning rate for generator
#LR_D = 0.0001           # learning rate for discriminator
LR_D = 0.001           # learning rate for discriminator
# 3,5,15
N_GEN_RANDINPUTS = 15


with tf.variable_scope('Gn'):
    # todo: conv2d
    G_in = tf.placeholder(tf.float32, [None, N_GEN_RANDINPUTS])          # random ideas (could from normal distribution)
    G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu)
    #G_output = tf.reshape(G_out1d, [-1, FLATTENED_SIZE])
    print("FLATTENED_SIZE", FLATTENED_SIZE)
    G_output = tf.layers.dense(G_l1, FLATTENED_SIZE)

    #G_output = tf.reshape(G_out1d, [] + list(SIZE_PIXELS))
    print('G_output', G_output)  #shape=(?, 20, 20, 3)

with tf.variable_scope('Discriminator'):
    real_art = tf.placeholder(tf.float32, [None,FLATTENED_SIZE], name='real_in')
    Discr_hiddenlayer_realinput = tf.layers.dense(real_art, 128, tf.nn.relu, name='l')

    #print('Discr_hiddenlayer_realinput', Discr_hiddenlayer_realinput)  #shape=(?, 20, 20, 128)
    #  WHERE is 3???
    Discr_out_realinput = tf.layers.dense(Discr_hiddenlayer_realinput, 1, tf.nn.sigmoid, name='out')              # probability that the art is real
    #print('*Discr_out_realinput', Discr_out_realinput)  # shape=(?, 20, 20, 1)

    # reuse layers for generator
    #Discr_hiddenlayer_fakeinput = tf.layers.dense(G_out1d, 128, tf.nn.relu, name='l', reuse=True)
    Discr_hiddenlayer_fakeinput = tf.layers.dense(G_output, 128, tf.nn.relu, name='l', reuse=True)
    #print('*Discr_hiddenlayer_fakeinput', Discr_hiddenlayer_fakeinput)
    #Discr_hiddenlayer_fakeinput = tf.layers.dense(G_output, 128, tf.nn.relu, name='l', reuse=True)            # receive art work from a newbie like G

    Discr_out_fakeinput = tf.layers.dense(Discr_hiddenlayer_fakeinput, 1, tf.nn.sigmoid, name='out', reuse=True)  # probability that the art work is made by artist

EPS =  +1e-30
D_loss = - tf.reduce_mean(tf.log(Discr_out_realinput + EPS) + tf.log(1-Discr_out_fakeinput + EPS))
G_loss =   tf.reduce_mean(                                    tf.log(1-Discr_out_fakeinput + EPS))

#D_loss = tf.Print(D_loss, [D_loss], "D_loss")
#G_loss = tf.Print(G_loss, [G_loss], "G_loss")

train_D = tf.train.AdamOptimizer(LR_D).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G = tf.train.AdamOptimizer(LR_G).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gn'))

start_time = time.time()


sess = tf.Session()

# For Tesnorboard
graph_writer = tf.summary.FileWriter("./graph/", sess.graph)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()




sess.run(init_op)

RESET_FRESH = True
session_saver = SessionSaver(sess, RESET_FRESH)


for step in range(5000*1000): #(500*1000):

    #if True or step == 0:
    if step == 0:
        artist_paintings_list = choose_random_batch(main_artworks, 64, FLATTENED_SIZE, RGB_CHANNELS)           # real painting from artist (15)

        actual_batchsize = len(artist_paintings_list)
        #print('££', len(artist_paintings_list), (artist_paintings_list[0].shape))
        #  artist_paintings_list: list of 20x20x3. list size = 64
        # intended: 64x20x20x3
        artist_paintings = np.stack(artist_paintings_list, axis=0)
        #print('££con:', artist_paintings.shape)

        #print(FLATTENED_SIZE, artist_paintings.shape[1:], artist_paintings.shape)
        assert FLATTENED_SIZE == artist_paintings.shape[1:]   # size: batchsize x arraysize


    G_randinput = np.random.randn(actual_batchsize, N_GEN_RANDINPUTS)

    G_paintings, pa0, Dl = sess.run([G_output, Discr_out_realinput, D_loss, train_D, train_G],    # train and get results
                                    {G_in: G_randinput, real_art: artist_paintings})[:3]

    if step % (2500) == 0:  # plotting

        print("step:", step,   "  last batchsize=", actual_batchsize, "  time (Sec):", time.time()-start_time)
        # for visualisation only:
        G_paintings2d = G_paintings[0,:].reshape(RGB_SIZE)
        PColor.plot_show_image(G_paintings2d, str(step), pa0, Dl)
        #plot_show_image(artist_paintings[0,:].reshape(RGB_SIZE), step)

        session_saver.tick(sess)

# For Tensorboard
graph_writer.close()

print("Finised. duration:", time.time()-start_time)
print("Time:", time.time())

PColor.last()
