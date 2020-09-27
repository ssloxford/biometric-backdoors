import argparse
import datetime
import json
import os
import socket
from os.path import join

import numpy as np
import pyperclip
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import plot_utils as put
from itertools import product

txt_filename = "transferability.tex"

label = "tab:transferability"

caption = """
Transferability results of the poisoning attack across different models.
The reported figures are the success rates of the attack uniquely using information from the source model (source is on the rows, target is on the columns).
Bold values refer to same-model success rates (also found in Table~
\\ref{tab:big_result}).
The rates correspond to the success at the 10\\textsuperscript{th} injection attempt.
The heuristic is always fit on the target system (i.e., both target model and classifier).
"""


if __name__ == "__main__":

    np.random.seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--destination", type=str)
    # parser.add_argument('-s', '--source_folder', action='store', required=True, choices=["vggresnet-vgg2_10", "facenet-vgg2_10", "vgg16-lfw_10"])
    # parser.add_argument('-du', '--n_dev_users', action='store', default=10)
    # parser.add_argument('-da', '--n_dev_attackers', action='store', default=10)
    parser.add_argument('-p', '--path_config', action='store', type=str, default=os.path.join("..", "face", "config.json"))

    args = parser.parse_args()

    # configuration file for face
    conf_face = json.load(open(args.path_config, "r"))
    np.set_printoptions(precision=4, suppress=True)

    clfs = ["cnt_flat", "mxm_flat", "lsvm_flat"]

    folders = [
        "facenet-vgg2_10",
        "facenet-casia_10",
        "vggresnet-vgg2_10",
        "vgg16-lfw_10",
    ]

    folder_pairs = list(product(folders, folders))
    print(folder_pairs)

    one_big_boy = {}

    for sf, tf in folder_pairs:
        if sf not in one_big_boy:
            one_big_boy[sf] = {}
        if tf not in one_big_boy[sf]:
            one_big_boy[sf][tf] = dict()
        conf_results = json.load(open(join(conf_face["paths"]["data"], "results", sf, "ccc.json"), "r"))
        victims = conf_results["victims"]
        adversaries = conf_results["adversaries"]
        selected_pairs = json.load(open(join(conf_face["paths"]["data"], "results", "selected_pairs.json"), "r"))
        adversaries = list(sorted(set([x for x, y in selected_pairs])))
        users = list(sorted(set([y for x, y in selected_pairs])))
        for clf in clfs:
            one_big_boy[sf][tf][clf] = [0, 2, 3]

        for clf in clfs:
            for atk, vic in selected_pairs:
                oof = join(conf_face["paths"]["data"], "results", sf, atk, vic, tf)
                if os.path.isfile(join(oof, "results.json")):
                    this_result = json.load(open(join(oof, "results.json"), "r"))
                    # 0 -> successful attack
                    # 2 -> failed because not enough samples are accepted at the end of the perturbations
                    # 3 -> failed because after 10 samples not enough adversary samples are accepted
                    if this_result["flat"][clf]["outcome"] in [0, 2, 3]:
                        one_big_boy[sf][tf][clf].append(this_result["flat"][clf]["outcome"])
                        pass

    output = "\\begin{table*}[t]\n"
    output += "\centering\n"
    output += "\\begin{tabular}{cccc|ccc|ccc|ccc}\n"
    output += "\\toprule\n"
    output += " & \multicolumn{3}{c}{%s} & \multicolumn{3}{c}{%s} & \multicolumn{3}{c}{%s} & \multicolumn{3}{c}{%s}\n" % (
        put.model_to_names[folders[0].split("_")[0]],
        put.model_to_names[folders[1].split("_")[0]],
        put.model_to_names[folders[2].split("_")[0]],
        put.model_to_names[folders[3].split("_")[0]],
    )
    output += "\\\\\n"
    output += "\\textit{Model} & \\texttt{centroid} & \\texttt{maximum} & \\texttt{SVM}& \\texttt{centroid} & \\texttt{maximum} & \\texttt{SVM}& \\texttt{centroid} & \\texttt{maximum} & \\texttt{SVM}& \\texttt{centroid} & \\texttt{maximum} & \\texttt{SVM} \\\\\midrule\n"

    # print(one_big_boy)
    my_failures = {}
    for sf in folders:
        #print(sf, sf.split("_")[0], put.model_to_names[sf.split("_")[0]])
        my_failures[sf] = {}
        
        output += put.model_to_names[sf.split("_")[0]] + " "
        for tf in folders:

            for clf in clfs:

                these_guys = np.array(one_big_boy[sf][tf][clf], dtype=int)

                bc = np.bincount(these_guys)
                percent_success = bc[0] / bc.sum()
                my_failures[sf][tf] = (bc[1] / bc.sum(), bc[2] / bc.sum())
                value_to_print = str("{0:.0f}".format(percent_success * 100))
                if tf != sf:
                    output += " & %s\%% " % value_to_print
                if tf == sf:
                    output += " & \\textbf{" + value_to_print + "}\% "

        output += "\\\\\n"
    output += "\\bottomrule\n"
    output += "\end{tabular}\n"
    output += "\caption{%s}\n" % caption
    output += "\label{%s}\n" % label
    output += "% Generated on {0:s} at {1:s}\n".format(socket.gethostname(), str(datetime.datetime.now()))
    output += "\end{table*}\n"
    print(output)
    pyperclip.copy(output)

    print(json.dumps(my_failures, sort_keys=True, indent=2))

    if args.destination is not None:
        print("[INFO] - creating %s" % os.path.join(args.destination, txt_filename))
        with open(os.path.join(args.destination, txt_filename), "w") as outfile:
            outfile.write(output)
        print("[INFO] - done")


