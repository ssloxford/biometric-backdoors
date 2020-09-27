import argparse
import datetime
import json
import os
import shutil
import socket
from os.path import join

import numpy as np
import pyperclip
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import plot_utils as put

def label_line(line, label, x, y, color='0.5', size=12):
    """Add a label to a line, at the proper angle.

    Arguments
    ---------
    line : matplotlib.lines.Line2D object,
    label : str
    x : float
        x-position to place center of text (in data coordinated
    y : float
        y-position to place center of text (in data coordinates)
    color : str
    size : float
    """
    xdata, ydata = line.get_data()
    x1 = xdata[0]
    x2 = xdata[-1]
    y1 = ydata[0]
    y2 = ydata[-1]

    ax = line.get_axes()
    text = ax.annotate(label, xy=(x, y), xytext=(-10, 0),
                       textcoords='offset points',
                       size=size, color=color,
                       horizontalalignment='left',
                       verticalalignment='bottom')

    sp1 = ax.transData.transform_point((x1, y1))
    sp2 = ax.transData.transform_point((x2, y2))

    rise = (sp2[1] - sp1[1])
    run = (sp2[0] - sp1[0])

    slope_degrees = np.degrees(np.arctan2(rise, run))
    text.set_rotation(slope_degrees)
    return text

if __name__ == "__main__":

    np.random.seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_config', action='store', type=str, default=os.path.join("..", "face", "config.json"))
    parser.add_argument("-d", "--destination", type=str)

    folders = ["facenet-vgg2_10", "vgg16-lfw_10"]

    args = parser.parse_args()

    # configuration file for face
    conf_face = json.load(open(args.path_config, "r"))
    np.set_printoptions(precision=4, suppress=True)

    one_big_boy = {}
    factors = ["age", "glasses", "beards", "pose", "poison", "random", "all"]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    line_objects = []
    line_obj_labels = []
    aucs = []

    markeverydict = {
        "age": 300,
        "pose": 150,
        "glasses": 50,
        "beards": 200
    }

    for i, f in enumerate(folders):
        conf_results = json.load(open(join(conf_face["paths"]["data"], "results", f, "ccc.json"), "r"))

        victims = conf_results["victims"]
        adversaries = conf_results["adversaries"]

        selected_pairs = json.load(open(join(conf_face["paths"]["data"], "results", "selected_pairs.json"), "r"))

        adversaries = list(sorted(set([x for x, y in selected_pairs])))
        users = list(sorted(set([y for x, y in selected_pairs])))

        # select n_dev attackers for development set
        # dev_adversaries = shuffle(adversaries)[:args.n_dev_attackers]

        dev_pairs = [[x, y] for x, y in selected_pairs]

        dev_data = {}
        for k in factors:
            dev_data[k] = []

        for atk, vic in dev_pairs:
            # check if file is there and is full
            oof = join(conf_face["paths"]["data"], "results", f, atk, vic, f)
            if os.path.isfile(join(oof, "counter.json")):
                this_counter = json.load(open(join(oof, "counter.json"), "r"))
                for k in factors:
                    if k == "all":
                        continue
                    # print(f, atk, vic)
                    dev_data[k].extend(this_counter["cnt_flat"]["cos_dist_%s"% k] )

        for k in factors:
            dev_data["all"].extend(dev_data[k])

        far_lim = 0.10

        fpr, tpr, thr, fnr, roc_auc, eer_thr_dev, eer_dev, arg_of_twop_far, far_1pc_thr = {}, {}, {}, {}, {}, {}, {}, {}, {}
        print("%s -> %s" % (f, f))

        # this loads the dev data
        for j, k in enumerate(factors):
            if k in ["poison", "all", "random"]:
                continue
            labels = np.hstack((np.ones(shape=len(dev_data["poison"])), np.zeros(shape=len(dev_data[k]))))
            preds = np.hstack((dev_data["poison"], dev_data[k]))

            fpr[k], tpr[k], thr[k] = roc_curve(labels, preds, pos_label=1)
            this_auc = roc_auc_score(labels, preds)
            aucs.append(this_auc)
            
            legend_label = k if i == 1 else ""

            if legend_label == "beards":
                legend_label = "facial hair"
            print(i, legend_label, this_auc)

            frr = 1-tpr[k]

            l_o = ax[i].plot(fpr[k], frr, label=legend_label, c=put.colors[j], lw=2,
                             marker=put.markers[j], markersize=6, markeredgewidth=1, markeredgecolor="k",markevery=markeverydict[k])
            if i ==1:
                line_objects.append(l_o)
                line_obj_labels.append(legend_label)
            margin = 0.00
            xlim = (0.0, .3)
            ylim = (0.0, .3)

        ax[i].set_xticks(np.arange(xlim[0], xlim[1]+.1, .1))
        ax[i].set_xlim(xlim[0] - margin, xlim[1])
        ax[i].set_yticks(np.arange(ylim[0], ylim[1]+.1, .1))
        ax[i].set_ylim(ylim[0], ylim[1] + margin)
        ax[i].set_xlabel("false accept rate")
        ax[i].set_ylabel("false reject rate")
        ax[i].set_title(put.model_to_names[f.split("_")[0]])

        eer_line,  = ax[i].plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], "--k", lw=2, label=None)
        #  = ax[i].annotate("\\textsc{eer}", [0.3, 0.8])

        ax[i].grid(True)

    fig.tight_layout(rect=(0, 0.04, 1.0, 1))
    # label_line(eer_line, "Some Label", .3, .8, color="k")
    line_obj_labels_auc = []
    for i, x in enumerate(line_obj_labels):
        val = "{0:0.2f}".format(aucs[i])
        val = val[1:]  # remove initial zero
        val = "%s (\\textsc{auc}=%s)" % (x, val)
        line_obj_labels_auc.append(val)
    # line_obj_labels = ["{0}({1:0.2f})".format(x, aucs[i]) for i, x in enumerate(line_obj_labels)]

    # Create the legend
    fig.legend(
        line_objects,  # The line objects
        labels=line_obj_labels,  # The labels for each line
        loc="lower center",  # Position of legend
        borderaxespad=0.05,  # Small spacing around legend box
        # title="Legend Title"  # Title for the legend
        ncol=4,
        fancybox=True,
        framealpha=.75,
        fontsize=24,
        handletextpad=.3,
        columnspacing=1
    )

    plt.savefig("./plots/countermeasure_roc.pdf", bbox_inches="tight")
    plt.show()

    if args.destination is not None:
        print("[INFO] - saving plots to %s" % os.path.join(args.destination, ))
        shutil.copy(
            os.path.join(".", "plots", "countermeasure_roc.pdf"),
            os.path.join(args.destination, "countermeasure_roc.pdf")
        )
