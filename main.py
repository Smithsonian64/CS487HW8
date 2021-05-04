import sys
import os
import numpy as np
import struct

import tensorflow as tf
import Cnn


# Press the green button in the gutter to run the script.


def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        sys.exit(0)
    elif sys.argv[1] == 'mnist':
        """Load MNIST data from `path`"""
        path = ''
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

        cnn = Cnn.cnn(data)




if __name__ == '__main__':
    main()
