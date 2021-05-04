import sys
import os
import numpy as np
import struct
import tensorflow as tf
import cnn


# Press the green button in the gutter to run the script.


def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        sys.exit(0)
    elif sys.argv[2] == 'mnist':
        """Load MNIST data from `path`"""
        path = '.'
        kind = 'train'
        labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

        with open(labels_path, 'rb') as lbpath:
                magic, n = struct.unpack('>II', lbpath.read(8))
                labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
                magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
                images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
                images = ((images / 255.) - .5) * 2
        data = (images, labels)




if __name__ == '__main__':
    x = tf.constant([[1., 2., 3., 4.],
                     [5., 6., 7., 8.],
                     [9., 10., 11., 12.]])
    x=tf.reshape(x, [256,32,27,27])
    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid')
    print(max_pool_2d(x))

    #main()
