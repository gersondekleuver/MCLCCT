from __future__ import division
import pickle
import matplotlib.pyplot as plt
import partition
import argparse
import numpy as np
import os
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import itertools
from sklearn.utils.extmath import cartesian


def plot_3D(round_n, epoch, number_of_trials=1, mode=0, number=0, single=0):

    dir = ""

    key = "/first_gen_"
    if mode == 1:
        key = "/generation_" + "0" + "_"

    elif mode == 2:
        key = "/last_generation_"

    with open(dir + key + str(1) + '.pkl', 'rb') as f:
        color_space = pickle.load(f)
        print(len(color_space.partitions))

    if single:

        agent = 0
        xs, ys, zs, color = [], [], [], []
        part = color_space.partitions[agent].partition
        points = color_space.partitions[agent].points

        # get keys
        for label in part:

            # loop over colors of label
            for pt in part[label]:
                point = points[pt]
                xs.append(point[0])
                ys.append(point[1])
                zs.append(point[2])
                color.append(label)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(xs, ys, zs, c=color, s=0.8)

        return plt.show()

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    xs, ys, zs, color = [], [], [], []
    part = color_space.partitions[0].partition
    points = color_space.partitions[0].points
    for label in part:
        for pt in part[label]:
            point = points[pt]
            xs.append(point[0])
            ys.append(point[1])
            zs.append(point[2])
            color.append(label)
    ax.scatter(xs, ys, zs, c=color, s=0.8)

    return plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--round_n", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--n_trials", type=float, default=1)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--number", type=str, default=0)
    parser.add_argument("--single", type=int, default=0)
    args = parser.parse_args()

    plot_3D(args.round_n, args.epochs, args.n_trials,
            args.mode, args.number, args.single)
