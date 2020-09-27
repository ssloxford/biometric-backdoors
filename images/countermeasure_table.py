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

label = "tab:countermeasure"

txt_filename = "countermeasure.tex"

caption = """
Rates for the poisoning detection. The top row indicates which model is considered for the evaluation: $a$\\textrightarrow$b$ indicates that the system is trained on $a$ and applied to $b$.
The values are computed on the \\texttt{centroid} with flat weights.
For same-model cases we report EER. For across-model cases we report the FAR and FRR for the detection on the target model obtained after choosing
the threshold at EER on the source model.
The ``all'' row is computed using all the updates from each other factor.
"""


def do_table(one_big_boy):
    fnvgg = "facenet-vgg2_10"
    vggrn = "vggresnet-vgg2_10"
    vgg16 = "vgg16-lfw_10"
    output = "\\begin{table*}[t]\n"
    output += "\centering\n"
    output += "\\begin{tabular}{cc|c|c|cc|cc|cc|cc}\n"
    output += "\\toprule\n"
    output += "& \multicolumn{1}{c}{%s} & \multicolumn{1}{c}{%s} & \multicolumn{1}{c}{%s} &\multicolumn{2}{c}{{\\footnotesize %s}} &\multicolumn{2}{c}{{\\footnotesize %s}} &\multicolumn{2}{c}{{\\footnotesize %s}} &\multicolumn{2}{c}{{\\footnotesize %s}} \\\\" % (
        put.model_to_names[fnvgg.split("_")[0]],
        put.model_to_names[vggrn.split("_")[0]],
        put.model_to_names[vgg16.split("_")[0]],
        put.model_to_names[fnvgg.split("_")[0]] + "\\textrightarrow " + put.model_to_names[vggrn.split("_")[0]],
        put.model_to_names[vggrn.split("_")[0]] + "\\textrightarrow " + put.model_to_names[fnvgg.split("_")[0]],
        put.model_to_names[fnvgg.split("_")[0]] + "\\textrightarrow " + put.model_to_names[vgg16.split("_")[0]],
        put.model_to_names[vgg16.split("_")[0]] + "\\textrightarrow " + put.model_to_names[vggrn.split("_")[0]],
    )
    output += "\\textit{factor} & "
    output += "\\textsc{eer} & \\textsc{eer} & \\textsc{eer} &"
    output += "\\textsc{far} & \\textsc{frr} & \\textsc{far} & \\textsc{frr} & \\textsc{far} & \\textsc{frr} & \\textsc{far} & \\textsc{frr} \\\\\midrule\n"

    def bf(inner):
        return "\\textbf{%s}" % inner

    for k in factors:
        if k == "poison" or k == "random":
            continue
        v1 = "{0:0.1f}\%".format(one_big_boy[(fnvgg, fnvgg)][k]["eer"] * 100)
        v2 = "{0:0.1f}\%".format(one_big_boy[(vggrn, vggrn)][k]["eer"] * 100)
        v3 = "{0:0.1f}\%".format(one_big_boy[(vgg16, vgg16)][k]["eer"] * 100)
        v4 = "{0:0.1f}\%".format(one_big_boy[(fnvgg, vggrn)][k]["far"] * 100)
        v5 = "{0:0.1f}\%".format(one_big_boy[(fnvgg, vggrn)][k]["frr"] * 100)
        v6 = "{0:0.1f}\%".format(one_big_boy[(vggrn, fnvgg)][k]["far"] * 100)
        v7 = "{0:0.1f}\%".format(one_big_boy[(vggrn, fnvgg)][k]["frr"] * 100)
        v8 = "{0:0.1f}\%".format(one_big_boy[(fnvgg, vgg16)][k]["far"] * 100)
        v9 = "{0:0.1f}\%".format(one_big_boy[(fnvgg, vgg16)][k]["frr"] * 100)
        v10 = "{0:0.1f}\%".format(one_big_boy[(vgg16, vggrn)][k]["far"] * 100)
        v11 = "{0:0.1f}\%".format(one_big_boy[(vgg16, vggrn)][k]["frr"] * 100)
        if k == "all":
            output += "%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            bf(k), bf(v1), bf(v2), bf(v3), bf(v4), bf(v5), bf(v6), bf(v7), bf(v8), bf(v9), bf(v10), bf(v11))
        else:
            output += "%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
                k if k != "beards" else "facial hair", v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11
            )

    output += "\\bottomrule\n"
    output += "\end{tabular}\n"
    output += "\caption{%s}\n" % caption
    output += "\label{%s}\n" % label
    output += "% Generated on {0:s} at {1:s}\n".format(socket.gethostname(), str(datetime.datetime.now()))
    output += "\end{table*}\n"
    print(output)
    pyperclip.copy(output)
    return output


if __name__ == "__main__":

    np.random.seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_config', action='store', type=str, default=os.path.join("..", "face", "config.json"))
    parser.add_argument("-d", "--destination", type=str)

    folder_pairs = [
        ["facenet-vgg2_10", "facenet-vgg2_10"],
        ["facenet-vgg2_10", "vggresnet-vgg2_10"],
        ["facenet-vgg2_10", "vgg16-lfw_10"],
        ["vggresnet-vgg2_10", "vggresnet-vgg2_10"],
        ["vggresnet-vgg2_10", "facenet-vgg2_10"],
        ["vgg16-lfw_10", "vgg16-lfw_10"],
        ["vgg16-lfw_10", "vggresnet-vgg2_10"]
    ]

    args = parser.parse_args()

    # configuration file for face
    conf_face = json.load(open(args.path_config, "r"))
    np.set_printoptions(precision=4, suppress=True)

    one_big_boy = {}
    factors = ["age", "glasses", "beards", "pose", "poison", "random", "all"]

    n_of_updates = {}

    for sf, tf in folder_pairs:
        one_big_boy[(sf, tf)] = {}
        conf_results = json.load(open(join(conf_face["paths"]["data"], "results", sf, "ccc.json"), "r"))

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
            oof = join(conf_face["paths"]["data"], "results", sf, atk, vic, sf)
            if os.path.isfile(join(oof, "counter.json")):
                this_counter = json.load(open(join(oof, "counter.json"), "r"))
                for k in factors:
                    if k == "all":
                        continue
                    dev_data[k].extend(this_counter["cnt_flat"]["cos_dist_%s"% k] )
                    #dev_data[k].extend(this_counter["mxm_flat"]["cos_dist_%s" % k])
                    #dev_data[k].extend(this_counter["lsvm_flat"]["cos_dist_%s" % k])

        for k in factors:
            dev_data["all"].extend(dev_data[k])

        far_lim = 0.10

        fpr, tpr, thr, fnr, roc_auc, eer_thr_dev, eer_dev, arg_of_twop_far, far_1pc_thr = {}, {}, {}, {}, {}, {}, {}, {}, {}
        print("%s -> %s" % (sf, tf))

        # this loads the dev data
        for k in factors:
            if k == "poison":
                continue
            labels = np.hstack((np.ones(shape=len(dev_data["poison"])), np.zeros(shape=len(dev_data[k]))))
            preds = np.hstack((dev_data["poison"], dev_data[k]))
            n_of_updates[k] = preds.shape[0]
            fpr[k], tpr[k], thr[k] = roc_curve(labels, preds, pos_label=1)
            roc_auc[k] = auc(fpr[k], tpr[k])
            fnr[k] = 1 - tpr[k]
            eer_dev[k] = brentq(lambda x: 1. - x - interp1d(fpr[k], tpr[k])(x), 0., 1.)
            eer_thr_dev[k] = interp1d(fpr[k], thr[k])(eer_dev[k])

            arg_of_twop_far[k] = np.where(fpr[k] >= far_lim)[0][0]
            # print(arg_of_twop_far)
            # print("far", fpr[k])
            # print("frr", fnr)

        print(n_of_updates)
        #print(eer_dev)
        #print(eer_thr_dev)

        # print info for same source-target
        if sf == tf:
            for k in factors:
                if k == "poison":
                    continue
                one_big_boy[(sf, tf)][k] = {}
                one_big_boy[(sf, tf)][k]["eer"] = eer_dev[k]
                one_big_boy[(sf, tf)][k]["frr@%dfar" % (far_lim*100)] = fnr[k][arg_of_twop_far[k]]
                print("{0}\t, eer {1:0.3f}, {2:0.5f} frr@{5:.0f}%far {3:0.3f}, {4:0.3f}".format(
                    k, eer_dev[k], eer_thr_dev[k], fnr[k][arg_of_twop_far[k]], thr[k][arg_of_twop_far[k]], far_lim*100
                ))
        else:
            # load from source but test on target
            tar_data = {}
            factors = ["age", "glasses", "beards", "pose", "poison", "random", "all"]
            for k in factors:
                tar_data[k] = []

            for atk, vic in dev_pairs:
                # check if file is there and is full
                oof = join(conf_face["paths"]["data"], "results", tf, atk, vic, tf)
                if os.path.isfile(join(oof, "counter.json")):
                    this_counter = json.load(open(join(oof, "counter.json"), "r"))
                    for k in factors:
                        if k == "all":
                            continue
                        tar_data[k].extend(this_counter["cnt_flat"]["cos_dist_%s" % k])
            for k in factors:
                tar_data["all"].extend(dev_data[k])

            for k in factors:
                if k == "poison":
                    continue
                # do predictions
                tp_mask = tar_data["poison"] >= eer_thr_dev[k]
                tp = tp_mask.astype(int).sum()
                fn_mask = tar_data["poison"] < eer_thr_dev[k]
                fn = fn_mask.astype(int).sum()
                fp_mask = tar_data[k] >= eer_thr_dev[k]
                fp = fp_mask.astype(int).sum()
                tn_mask = tar_data[k] < eer_thr_dev[k]
                tn = tn_mask.astype(int).sum()
                far = fp/(fp+tn)
                frr = fn/(fn+tp)
                # print(fp, tp, fn, tn)
                print("{0}\t, thr {1:0.3f} far {2:0.3f} frr {3:0.3f}".format(
                    k, thr[k][arg_of_twop_far[k]], far, frr
                ))
                one_big_boy[(sf, tf)][k] = {}
                one_big_boy[(sf, tf)][k]["far"] = far
                one_big_boy[(sf, tf)][k]["frr"] = frr
        print("")

    output = do_table(one_big_boy)

    if args.destination is not None:
        print("[INFO] - creating %s" % os.path.join(args.destination, txt_filename))
        with open(os.path.join(args.destination, txt_filename), "w") as outfile:
            outfile.write(output)
        print("[INFO] - done")

