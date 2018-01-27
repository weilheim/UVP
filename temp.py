from __future__ import print_function

import h5py
import matplotlib.pyplot as plt
import numpy as np

data_path = 'data/mnist_test_seq.npy'
data = np.load(data_path)
print(data.shape)

seq = data[:, 0, :, :]
seq_len = seq.shape[0]
plt.figure(num=0, figsize=(seq_len, 1))
plt.clf()
for i in xrange(seq_len):
    plt.subplot(1, seq_len, i+1)
    plt.imshow(seq[i, :, :], cmap=plt.cm.gray, interpolation='bilinear')
    plt.axis('off')
plt.draw()
plt.pause(2)