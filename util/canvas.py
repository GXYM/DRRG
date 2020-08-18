#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = '古溪'

import numpy as np
import random
import matplotlib.pyplot as plt


def heatmap(im_gray):
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(255 - im_gray)
    Hmap = np.delete(rgba_img, 3, 2)
    # print(Hmap.shape, Hmap.max(), Hmap.min())
    # cv2.imshow("heat_img", Hmap)
    # cv2.waitKey(0)
    return Hmap


def loss_ploy(loss_list, steps, period, name=""):
    fig1, ax1 = plt.subplots(figsize=(16, 9))
    ax1.plot(range(steps // period), loss_list)
    ax1.set_title("Average loss vs step*{}".format(period))
    ax1.set_xlabel("step*{}".format(period))
    ax1.set_ylabel("Current loss")
    plt.savefig('{}@loss_vs_step*{}.png'.format(name,period))
    plt.clf()


def plt_ploys(ploys, period, name=""):
    fig1, ax1 = plt.subplots(figsize=(16, 9))
    cnames = ['aliceblue','antiquewhite','aqua','aquamarine','azure',
               'blanchedalmond','blue','blueviolet','brown','burlywood',
               'coral','cornflowerblue','cornsilk','crimson','cyan',
               'darkblue','deeppink','deepskyblue','dodgerblue','forestgreen',
               'gold','goldenrod','green','greenyellow','honeydew','hotpink',
               'lawngreen','lightblue','lightgreen','lightpink','lightsalmon',
               'lightseagreen','lightsteelblue','lightyellow','lime','limegreen',
               'mediumseagreen','mediumspringgreen','midnightblue','orange','orangered',
               'pink','red','royalblue','seagreen','skyblue','springgreen','steelblue',
               'tan','teal','thistle','yellow','yellowgreen']

    color = random.sample(cnames, len(ploys.keys()))
    for ii, key in enumerate(ploys.keys()):
        ax1.plot(range(1, len(ploys[key])+1), ploys[key],color=color[ii], label=key)
    ax1.set_title("Loss Carve line")
    ax1.set_xlabel("step*{}".format(period))
    ax1.set_ylabel("Current loss")
    plt.legend(ploys.keys())
    plt.savefig('{}@loss_vs_step*{}.png'.format(name, period))
    plt.clf()

if __name__ == '__main__':
    # TODO ADD CODE
    pass