import argparse
import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
import sys
import matplotlib
from os.path import join
from face import face_utils as fut
import plot_utils as put

_x_ticks = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fontsize = 21
_x_lim = (-0.1, 10.1)
_y_lim = (-0.05, 1.05)
_ci = 0.99


def get_rates(source_folder, target_folder, adversaries, victims):
    iar = {}
    far = {}
    frr = {}
    for a in adversaries:
        for v in victims:
            results_filepath = join(source_folder, a, v, target_folder, "results.json")
            if not os.path.isfile(results_filepath):
                continue
            results = json.load(open(results_filepath, "r"))
            for w in ["flat", "sigmoid"]:
                for c in ["cnt", "mxm", "lsvm"]:
                    if results[w]["%s_%s" % (c, w)]["outcome"] == 0:
                        assert len(results[w]["%s_%s" % (c, w)]["iar"]) == 11
                        assert len(results[w]["%s_%s" % (c, w)]["frr"]) == 11
                        assert len(results[w]["%s_%s" % (c, w)]["far"]) == 11
                        if "%s_%s" % (c, w) not in iar:
                            iar["%s_%s" % (c, w)] = []
                        if "%s_%s" % (c, w) not in frr:
                            frr["%s_%s" % (c, w)] = []
                        if "%s_%s" % (c, w) not in far:
                            far["%s_%s" % (c, w)] = []

                        iar["%s_%s" % (c, w)].append(results[w]["%s_%s" % (c, w)]["iar"])
                        frr["%s_%s" % (c, w)].append(results[w]["%s_%s" % (c, w)]["frr"])
                        far["%s_%s" % (c, w)].append(results[w]["%s_%s" % (c, w)]["far"])

    return iar, far, frr


if __name__ == "__main__":

    np.random.seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--destination", action="store", type=str)
    # parser.add_argument('-t', '--target_folder', action='store', required=True)
    parser.add_argument('-p', '--path_config', action='store', type=str, default=os.path.join("..", "face", "config.json"))

    args = parser.parse_args()

    # configuration file for face
    conf_face = json.load(open(args.path_config, "r"))
    conf_results = json.load(open(join(conf_face["paths"]["data"], "results", "facenet-vgg2_10", "ccc.json"), "r"))

    victims = conf_results["victims"]
    adversaries = conf_results["adversaries"]
    test_users = list(conf_results["test_users"].keys())

    folders = ["facenet-vgg2_10", "vggresnet-vgg2_10", "vgg16-lfw_10"]

    rows = [put.model_to_names[x.split("_")[0]] for x in folders]
    cols = ["\\texttt{centroid}", "\\texttt{maximum}", "\\texttt{SVM}"]

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(16, 10))

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], rows):
        ax.set_ylabel(row, size='large')

    for i, folder in enumerate(folders):

        sf = folder[:-3]
        tf = folder[:-3]

        source_folder = join(conf_face["paths"]["data"], "results", folder)
        target_folder = folder

        model_name = folder.split("_")[0]
        models_folder = join(conf_face["paths"]["data"], "models")

        meta_dir = fut.get_meta_folder(config=conf_face, model_name=model_name)
        meta_dir = os.path.join(conf_face["paths"]["data"], "dataset", meta_dir)

        iar, far, frr = get_rates(source_folder, target_folder, adversaries, victims)
        w = "flat"

        for j, c in enumerate(["cnt", "mxm", "lsvm"]):
            this_iar = np.array(iar[c+"_"+w])
            this_frr = 1-np.array(frr[c+"_"+w])
            this_far = np.array(far[c+"_"+w])
            n_samples = 11
            cols = range(0, n_samples)
            xs = cols

            print("%s, %s, %s, # lines: %d" % (folder, w, c, this_iar[:, cols].shape[0]))

            y_iar = this_iar[:, cols].mean(axis=0)
            y_frr = this_frr[:, cols].mean(axis=0)
            y_far = this_far[:, cols].mean(axis=0)

            mins_fa, maxs_fa = [], []
            mins_fo, maxs_fo = [], []
            mins_fu, maxs_fu = [], []
            for jj in range(n_samples):
                _, a, b = put.mean_confidence_interval(this_iar[:, jj], confidence=_ci)
                mins_fa.append(a)
                maxs_fa.append(b)
                _, a, b = put.mean_confidence_interval(this_far[:, jj], confidence=_ci)
                mins_fo.append(a)
                maxs_fo.append(b)
                _, a, b = put.mean_confidence_interval(this_frr[:, jj], confidence=_ci)
                mins_fu.append(a)
                maxs_fu.append(b)

            l1 = axes[i, j].plot(xs, y_iar, '-^', c=put.colors[0], linewidth=2, marker=put.markers[0], markersize=6, markeredgewidth=1,
                 markeredgecolor="k", label="IAR", zorder=1, )[0]
            l2 = axes[i, j].plot(xs, y_frr, '-^', c=put.colors[1], linewidth=2, marker=put.markers[1], markersize=6, markeredgewidth=1,
                 markeredgecolor="k", label="FRR", zorder=1, )[0]
            l3 = axes[i, j].plot(xs, y_far, '-^', c=put.colors[2], linewidth=2, marker=put.markers[2], markersize=6, markeredgewidth=1,
                 markeredgecolor="k", label="FAR", zorder=1, )[0]

            axes[i, j].fill_between(xs, mins_fa, maxs_fa, alpha=0.5, edgecolor=put.colors[0], facecolor=put.colors[0],
                             linewidth=0,
                             zorder=1)

            axes[i, j].fill_between(xs, mins_fu, maxs_fu, alpha=0.5, edgecolor=put.colors[1], facecolor=put.colors[1],
                             linewidth=0,
                             zorder=1)

            axes[i, j].fill_between(xs, mins_fo, maxs_fo, alpha=0.5, edgecolor=put.colors[2], facecolor=put.colors[2],
                             linewidth=0,
                             zorder=1)

            axes[i, j].set_xticks(_x_ticks)
            axes[i, j].set_xticklabels(_x_ticks, fontsize=fontsize)
            axes[i, j].set_xlim(_x_lim)
            axes[i, j].set_ylim(_y_lim)
            axes[i, j].set_xlabel("\# of injection attempts", fontsize=fontsize)
            # axes[i, j].ylabel("rate", fontsize=fontsize)
            legend_loc = "center right"
            # axes[i, j].legend(loc=legend_loc, fontsize=fontsize - 2)
            axes[i, j].grid()
            axes[i, j].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            axes[i, j].set_yticklabels(["0.0", ".25", ".50", ".75", "1.0"], fontsize=fontsize)
            # plt.savefig("./plots/rates_%s_%s_%s_%s.pdf" % (sf, tf, c, w), bbox_inches="tight")
    fig.tight_layout(rect=(0,0,1.0,.96))

    # Create the legend
    fig.legend(
        [l1, l2, l3],  # The line objects
        labels=["IAR", "FRR", "FAR"],  # The labels for each line
        loc="upper center",  # Position of legend
        borderaxespad=0.05,  # Small spacing around legend box
        # title="Legend Title"  # Title for the legend
        ncol=3
    )

    plt.savefig("./plots/rates.pdf", bbox_inches="tight")

    if args.destination is not None:
        print("[INFO] - saving plots to %s" % os.path.join(args.destination,))
        shutil.copy(
            os.path.join(".", "plots", "rates.pdf"),
            os.path.join(args.destination, "rates.pdf")
        )






