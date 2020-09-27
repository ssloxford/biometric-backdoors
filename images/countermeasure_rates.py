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


def get_rates(source_folder, target_folder, adversaries, victims):
    pres = []
    posts = []
    for a in adversaries:
        for v in victims:
            results_filepath = join(source_folder, a, v, target_folder, "applied_cm.json")
            if not os.path.isfile(results_filepath):
                continue
            results = json.load(open(results_filepath, "r"))
            clf_n = "cnt_flat"

            if len(results[clf_n]["iar_pre"]) != 11 or len(results[clf_n]["iar_post"]) != 11:
                continue
            else:
                pres.append(results[clf_n]["iar_pre"])
                posts.append(results[clf_n]["iar_post"])

    return pres, posts


if __name__ == "__main__":

    np.random.seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--destination", action="store", type=str)
    parser.add_argument('-p', '--path_config', action='store', type=str, default=os.path.join("..", "face", "config.json"))

    args = parser.parse_args()

    # configuration file for face
    conf_face = json.load(open(args.path_config, "r"))
    conf_results = json.load(open(join(conf_face["paths"]["data"], "results", "facenet-vgg2_10", "ccc.json"), "r"))

    victims = conf_results["victims"]
    adversaries = conf_results["adversaries"]
    test_users = list(conf_results["test_users"].keys())

    folders = ["facenet-vgg2_10", "vggresnet-vgg2_10", "vgg16-lfw_10"]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))

    for i, f in enumerate(folders):

        pre, post = get_rates(os.path.join(conf_face["paths"]["data"], "results", f), f, adversaries, victims)

        pre = np.array(pre)
        post = np.array(post)

        this_color = put.colors[i]

        # count how many times attack is detected at 1-2-3
        p1 = post[:, 1]
        p2 = post[:, 2]
        p3 = post[:, 3]
        print(post.shape, (p3>p2).astype(int).sum())
        # print(post[:, 1], post[:,2])

        print(f, pre.shape[0], "# attacks that survive the 2nd injection: %.2f%%" % ((p3>p2).astype(int).sum()/post.shape[0]*100))

        plt.plot(range(0, 11), pre.mean(axis=0), c=this_color, marker=put.markers[i], lw=2, linestyle="dashed", ms=8)
        plt.plot(range(0, 11), post.mean(axis=0), c=this_color, marker=put.markers[i], lw=2, label=put.model_to_names[f.split("_")[0]], ms=8)

    plt.ylim(-0.0, 0.2)
    plt.xlim(-.05, 4.05)
    plt.xticks(range(0, 5), range(0, 5))
    plt.xlabel("\# of injection attempts")
    plt.ylabel("IAR")
    plt.legend(loc="upper left", fancybox=True, framealpha=.75)
    plt.grid()

    plt.savefig("./plots/countermeasure_rates.pdf", bbox_inches="tight")
    plt.show()

    if args.destination is not None:
        print("[INFO] - saving plots to %s" % os.path.join(args.destination, ))
        shutil.copy(
            os.path.join(".", "plots", "countermeasure_rates.pdf"),
            os.path.join(args.destination, "countermeasure_rates.pdf")
        )
