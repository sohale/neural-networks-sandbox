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
            self.saver.restore(sess, self.SAVED_SESSION_FILENAME)

    def tick(self, sess):
        self.save_path = self.saver.save(sess, self.SAVED_SESSION_FILENAME)

"""
from mnist_works import loadmnist_from_args
img, label, RESET_FRESH = loadmnist_from_args()
"""


exper_params = {
    'data_samples': 1000,
    'seed1': 1,
    'seed2': 1,
    'train_iters': 5000*1000,  # (500*1000)
}

# Hyper Parameters
#  IMPORTANT DESIGN CHOICES

hyperparams = {

    'w': 14,
    'h': 14,
    'rgb_channels': 1,

    'Gn_inputs': 3,  # 3,5,15
    'Gn_layers': [None, 8,16,32,64, 128, None],
    'Dc_layers': [None, 128, 64, 32, 16, 8, None],

    'pixel_dtype': tf.float32,

    'Gn_input_dtype': tf.float32,

    'batch_size': 64*5,  # 64

    'LearningRate_Gn': 0.0001, # 0.001
    'LearningRate_Dc': 0.0001, # 0.001

    'Gn_input_distr': 'randu',  # 'randn'


    'eps':  +1e-30,

}

RGB_CHANNELS = hyperparams['rgb_channels']

# IMPORTANT DESIGN CHOICE
# 7, 14, 20, 28, 200
RGB_SIZE = (hyperparams['w'], hyperparams['h'], RGB_CHANNELS)

FLATTENED_SIZE = np.prod(np.array(RGB_SIZE))

BATCHSIZE_PROV = hyperparams['batch_size']

HOW_MANY_SAMPLES_SYNTHESIZED = exper_params['data_samples']
# img, label, RESET_FRESH = simple_traiangles()
main_dataset = simple_triangles(FLATTENED_SIZE/RGB_CHANNELS, RGB_CHANNELS, (RGB_SIZE[0],RGB_SIZE[1]), HOW_MANY_SAMPLES_SYNTHESIZED)
print('synthesized %d samples' % HOW_MANY_SAMPLES_SYNTHESIZED)


def rand_generator(rows, cols):
    if hyperparams['Gn_input_distr'] == 'randn':
        return np.random.randn(rows, cols)
    if hyperparams['Gn_input_distr'] == 'randu':
        return np.random.rand(rows, cols)
    else:
        raise Exception('unknown')

PColor.init()

tf.set_random_seed(exper_params['seed1'])
np.random.seed(exper_params['seed2'])

EPS = hyperparams['eps']

LearningRate_Gn = hyperparams['LearningRate_Gn']          # learning rate for generator
LearningRate_Dc = hyperparams['LearningRate_Dc']           # learning rate for discriminator
N_GEN_RANDINPUTS = hyperparams['Gn_inputs']


#Gn_L1 = hyperparams['Gn_layers'][1]
#Dc_L1 = hyperparams['Dc_layers'][1]

"""
    @param `layers_array` is [None, 128, None]
    e.g.: hyperparams['Gn_layers']
"""
def layersize_iterator(layers_array):
    assert layers_array[0] is None
    assert layers_array[-1] is None
    assert len(layers_array) >= 2
    hidden_layersz = layers_array[1:-1]
    print('hidden_layersz', hidden_layersz)
    for i in range(len(hidden_layersz)):
        hlsize = hidden_layersz[i]
        yield i, hlsize


DCR_OUTPUTS = 1

print(
    'Report:\n', '*'*70,
    '\n',
    'Layers:',
    '\n       Gn:',
    [N_GEN_RANDINPUTS, hyperparams['Gn_layers'][1:-1], FLATTENED_SIZE],
    '\n       Dc:',
    [FLATTENED_SIZE, hyperparams['Dc_layers'][1:-1], DCR_OUTPUTS],
    '\n', '*'*70,
)

with tf.variable_scope('Gn'):
    # todo: conv2d
    Gn_input_layer = tf.placeholder(hyperparams['Gn_input_dtype'], [None, N_GEN_RANDINPUTS])          # (from normal distribution)
    Gn_hidden_layer = Gn_input_layer
    for i, Gn_hlsize in layersize_iterator(hyperparams['Gn_layers']):
        Gn_hidden_layer = tf.layers.dense(Gn_hidden_layer, Gn_hlsize, tf.nn.relu)
    #Gn_output_layer = tf.reshape(G_out1d, [-1, FLATTENED_SIZE])
    print("FLATTENED_SIZE", FLATTENED_SIZE)
    Gn_output_layer = tf.layers.dense(Gn_hidden_layer, FLATTENED_SIZE)

    #Gn_output_layer = tf.reshape(G_out1d, [] + list(SIZE_PIXELS))
    print('Gn_output_layer', Gn_output_layer)  #shape=(?, 20, 20, 3)

with tf.variable_scope('Dc'):
    real_input = tf.placeholder(hyperparams['pixel_dtype'], [None,FLATTENED_SIZE], name='real_in')

    # real versus fake inputs
    Dc_hiddenlayer_real = real_input
    Dc_hiddenlayer_fake = Gn_output_layer
    for i, Dc_hlsize in layersize_iterator(hyperparams['Dc_layers']):
        # Dc_L1
        lname = 'Dc_h'+str(i)
        Dc_hiddenlayer_real = tf.layers.dense(Dc_hiddenlayer_real, Dc_hlsize, tf.nn.relu, name=lname)
        Dc_hiddenlayer_fake = tf.layers.dense(Dc_hiddenlayer_fake, Dc_hlsize, tf.nn.relu, name=lname, reuse=True)
        #Dc_hiddenlayer_fake = tf.layers.dense(G_out1d, Dc_L1, tf.nn.relu, name='Dc_h1', reuse=True)
        #Dc_hiddenlayer_fake = tf.layers.dense(Gn_output_layer, Dc_L1, tf.nn.relu, name='Dc_h1', reuse=True)            # receive art work from a newbie like G
        #print('*Dc_hiddenlayer_fake', Dc_hiddenlayer_fake)

    #print('Dc_hiddenlayer_real', Dc_hiddenlayer_real)  #shape=(?, 20, 20, Dc_L1)
    #  WHERE is 3???
    Dc_out_realinput = tf.layers.dense(Dc_hiddenlayer_real, DCR_OUTPUTS, tf.nn.sigmoid, name='Dc_out')              # probability that the image is genuine/real
    Dc_out_fakeinput = tf.layers.dense(Dc_hiddenlayer_fake, DCR_OUTPUTS, tf.nn.sigmoid, name='Dc_out', reuse=True)  # probability that the image is genuine/real
    #print('*Dc_out_realinput', Dc_out_realinput)  # shape=(?, 20, 20, 1)


D_loss = - tf.reduce_mean(tf.log(Dc_out_realinput + EPS) + tf.log(1-Dc_out_fakeinput + EPS))
G_loss =   tf.reduce_mean(                                 tf.log(1-Dc_out_fakeinput + EPS))

#D_loss = tf.Print(D_loss, [D_loss], "D_loss")
#G_loss = tf.Print(G_loss, [G_loss], "G_loss")

train_D = tf.train.AdamOptimizer(LearningRate_Dc).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Dc'))
train_G = tf.train.AdamOptimizer(LearningRate_Gn).minimize(
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

for step in range(exper_params['train_iters']):

    #if True or step == 0:
    if step == 0:
        images_batch__list = choose_random_batch(main_dataset, BATCHSIZE_PROV, FLATTENED_SIZE, RGB_CHANNELS, True)

        actual_batchsize = len(images_batch__list)
        #print('££', len(images_batch__list), (images_batch__list[0].shape))
        #  images_batch__list: list of 20x20x3. list size = 64
        # intended: 64x20x20x3
        images_training_batch = np.stack(images_batch__list, axis=0)
        #print('££con:', images_training_batch.shape)

        #print(FLATTENED_SIZE, images_training_batch.shape[1:], images_training_batch.shape)
        assert FLATTENED_SIZE == images_training_batch.shape[1:]   # size: batchsize x arraysize


    G_randinput = rand_generator(actual_batchsize, N_GEN_RANDINPUTS)

    G_paintings, pa0, Dl = sess.run([Gn_output_layer, Dc_out_realinput, D_loss, train_D, train_G],    # train and get results
                                    {Gn_input_layer: G_randinput, real_input: images_training_batch})[:3]

    if step % (2500) == 0:  # plotting

        print("step:", step,   "  last batchsize=", actual_batchsize, "  time (Sec):", time.time()-start_time)
        # for visualisation only:
        G_paintings2d = G_paintings[0,:].reshape(RGB_SIZE)
        #print(G_paintings2d.shape, "shape<<<<", np.max(G_paintings2d.ravel()), G_paintings2d.dtype)

        PColor.plot_show_image(
            G_paintings2d,
            'generated-' + str(step),
            0.1,
            [pa0.mean(), -Dl]
        )
        if step == 0:
            for data_idx in range(0,15):
                PColor.plot_show_image(
                    images_training_batch[data_idx,:].reshape(RGB_SIZE),
                    'train-' + str(data_idx)+'@'+str(step),
                    0.1,
                    [0,0]
                )

        session_saver.tick(sess)

# For Tensorboard
graph_writer.close()

print("Finised. duration:", time.time()-start_time)
print("Time:", time.time())

PColor.last()
