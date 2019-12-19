notes.md


## Run on MacOS


MacOS:
Installation on MacOS: (First time only)
    * virtualenv --version # If error, install virsualenv . see https://www.tensorflow.org/install/pip
    * cd glyphnet
    * virtualenv -v --python=python3  ./tensorf1
    * source ./tensorf1/bin/activate
    * pip install tensorflow==1.15.0
    * pip install scipy
    * pip install imageio
    * pip install  matplotlib
    * pip install scikit-image

    Unsure: cython PyHamcrest

Run on MacOS
    * cd glyphnet
    * source ./tensorf1/bin/activate
    * python glyphnet1.py


## Run on Linux
MNIST only (unused)
```bash
PYTHONPATH=. python ./bin/mnistgan.py
```

## Run on Windows (Anaconda)


## Tensorboard
 1. Add two lines to code.
    ```python
        graph_writer = tf.summary.FileWriter("./graph/", sess.graph)
        graph_writer.close()
    ```
 2. Run in commandline:   `tensorboard --logdir="./graph"`
 3. Browse:  `http://localhost:6006/`
See: https://www.tensorflow.org/guide/graphs

## Misc notes
Based on:
https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/406_GAN.py

### delete:
tf.layers.conv2d
input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
[batch_size, image_height, image_width, channels]
print(tf.__file__)


Read misc:
CNN:
https://www.tensorflow.org/tutorials/estimators/cnn
GAN + conv2d:
https://datascience.stackexchange.com/questions/30810/gan-with-conv2d-using-tensorflow-shape-error
morvanzhou:
https://morvanzhou.github.io/tutorials/
https://www.youtube.com/user/MorvanZhou





linux preparation (ubuntu 16)
virtualenv -v --python=python3  ./tensorf2
pip install tensorflow
(failed)

virtualenv -v --python=python3  ./tensorf1
pip install tensorflow==1.15.0


pip install scipy
pip install imageio
pip install matplotlib
pip install scikit-image
