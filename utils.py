from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np


def show_video(video, pause=3):
    """show video data
    :param video: 3d np array: vid_len x image_size x image_size
    :return:
    """
    assert len(video.shape) == 3
    vid_len = video.shape[0]
    plt.figure(num=0, figsize=(vid_len, 1))
    plt.clf()
    for i in xrange(vid_len):
        plt.subplot(1, vid_len, i + 1)
        plt.imshow(video[i, :, :], cmap=plt.cm.gray, interpolation='bilinear')
        plt.axis('off')
    plt.draw()
    plt.pause(pause)
