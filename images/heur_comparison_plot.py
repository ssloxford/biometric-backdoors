import argparse
import sys, os
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import json
import scipy.stats as st


colors = [
    "b",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "grey",
    "black",
    "yellow",
    "aqua"
]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', action='store', required=True)  # tmp/5_15_5
    args = parser.parse_args()
    conf1 = json.load(open("../adversarial/config.json", "r"))
    conf2 = json.load(open(os.path.join(args.folder, "ccc.json"), "r"))

    dist_lin_bb = []
    dist_lin_bb2 = {}

    bb_lin_counter = 0
    bb2_lin_counter = {}

    bb_lin_counter_f = 0
    bb2_lin_counter_f = 0

    strategies = ["conservative_0.25_ss", "aggressive_0.15_ss"]

    bb2_si_lin = {}
    bb2_si_lin_all = {}

    for s in strategies:
        bb2_si_lin[s] = [0, 0]
        bb2_si_lin_all[s] = []
        dist_lin_bb2[s] = []
        bb2_lin_counter[s] = 0

    n_atts = {
        "conservative_0.25_ss": [],
        "aggressive_0.15_ss": [],
        "iterative_step_10": [],
        "iterative_step_1": []
    }

    n_atts_tot = {
        "conservative_0.25_ss": [],
        "aggressive_0.15_ss": [],
        "iterative_step_10": [],
        "iterative_step_1": []
    }

    f = "blackbox"

    adversaries = list(filter(lambda x: os.path.isdir(os.path.join(args.folder, f, x)), os.listdir(os.path.join(args.folder, f))))
    for a in adversaries:
        victims = list(filter(lambda x: os.path.isdir(os.path.join(args.folder, f, a, x)), os.listdir(os.path.join(args.folder, f, a))))
        raw_adv = np.load(os.path.join(conf1["adv_path"], a, "raw.npy"))
        for v in victims:
            targets = list(filter(lambda x: os.path.isdir(os.path.join(args.folder, f, a, v, x)),
                                  os.listdir(os.path.join(args.folder, f, a, v))))
            if len(targets) > 0:
                for t in targets:
                    if os.path.isfile(os.path.join(args.folder, f, a, v, t, "dist_after_injection_linear.csv")):
                        db = np.loadtxt(os.path.join(args.folder, f, a, v, t, "dist_after_injection_linear.csv"))
                        dist_lin_bb.append(db.tolist())
                        bb_lin_counter += 1
                    else:
                        bb_lin_counter_f += 1

                    if os.path.isfile(os.path.join(args.folder, f, a, v, t, "perturb_index_linear.csv")):
                        pil = np.loadtxt(os.path.join(args.folder, f, a, v, t, "perturb_index_linear.csv"), ndmin=1)
                        n_atts["iterative_step_10"].append(int(pil.sum() / 10))
                        n_atts["iterative_step_1"].append(int(pil.sum()))
                        n_atts_tot["iterative_step_10"].append(int(pil.sum() / 10) + pil.shape[0])
                        n_atts_tot["iterative_step_1"].append(int(pil.sum()) + pil.shape[0])

                    if os.path.isdir(os.path.join(args.folder, f, a, v, t, "bb")):
                        for strat in strategies:
                            dist_file = os.path.join(args.folder, f, a, v, t, "bb", "dist_after_injection_linear_%s.csv" % strat)
                            succ_file = os.path.join(args.folder, f, a, v, t, "bb", "successful_inj_linear_%s.csv" % strat)
                            if os.path.isfile(dist_file):
                                if os.path.isfile(succ_file):
                                    db = np.loadtxt(dist_file, ndmin=1)
                                    sil = np.loadtxt(succ_file, ndmin=1)
                                    bb2_si_lin_all[strat].append((sil * 2 - 1).tolist())
                                    kk = 0
                                    _l = [db[0]]
                                    for jj in range(sil.shape[0]):
                                        if sil[jj] == 0:
                                            _l.append(db[kk])
                                        else:
                                            kk += 1
                                            _l.append(db[kk])
                                    dist_lin_bb2[strat].append(_l)
                                    bb2_lin_counter[strat] += 1
                                    prev = 1
                                    for item in sil:
                                        if prev == 1 and item == 0:
                                            bb2_si_lin[strat][1] += 1
                                        if prev == 1 and item == 1:
                                            bb2_si_lin[strat][0] += 1
                                        prev = item

    N = 250

    print("blackbox %d/%d, %.2f %%" % (bb_lin_counter, bb_lin_counter_f+bb_lin_counter, bb_lin_counter/(bb_lin_counter+bb_lin_counter_f)))

    for s in strategies:
        print("===Strategy '%s'===" % s)
        print("blackbox2 %d/%d, %.2f %%" % (
            bb2_lin_counter[s],
            bb_lin_counter,
            bb2_lin_counter[s]/(bb_lin_counter)
        ))
        print("blackbox2 heuristic %d/%d %.2f %%" % (
            bb2_si_lin[s][0],
            bb2_si_lin[s][1] + bb2_si_lin[s][0],
            bb2_si_lin[s][0] / (bb2_si_lin[s][0] + bb2_si_lin[s][1])
        ))

    padded_lin_bb = np.zeros(shape=(len(dist_lin_bb), N))
    padded_lin_bb2 = {
        "aggressive_0.15_ss": np.zeros(shape=(len(dist_lin_bb2["aggressive_0.15_ss"]), N)),
        "conservative_0.25_ss": np.zeros(shape=(len(dist_lin_bb2["conservative_0.25_ss"]), N))
    }
    padded_si_bb2 = {
        "aggressive_0.15_ss": np.zeros(shape=(len(padded_lin_bb2["aggressive_0.15_ss"]), N)),
        "conservative_0.25_ss": np.zeros(shape=(len(padded_lin_bb2["conservative_0.25_ss"]), N))
    }

    for i in range(padded_lin_bb.shape[0]):
        if type(dist_lin_bb[i]) == float:
            dist_lin_bb[i] = [dist_lin_bb[i]]
        padded_lin_bb[i, 0:len(dist_lin_bb[i])] = np.array(dist_lin_bb[i])

    for s in strategies:
        for i in range(padded_lin_bb2[s].shape[0]):
            if type(dist_lin_bb2[s][i]) == float:
                dist_lin_bb2[s][i] = [dist_lin_bb2[s][i]]
            padded_lin_bb2[s][i, 0:len(dist_lin_bb2[s][i])] = np.array(dist_lin_bb2[s][i])
        for i in range(padded_si_bb2[s].shape[0]):
            if type(bb2_si_lin_all[s][i]) == float:
                bb2_si_lin_all[s][i] = [bb2_si_lin_all[s][i]]
            padded_si_bb2[s][i, 0:len(bb2_si_lin_all[s][i])] = np.array(bb2_si_lin_all[s][i])
        padded_si_bb2[s] = padded_si_bb2[s][:, ~np.all(padded_si_bb2[s]==0, axis=0)]
        padded_si_bb2[s][padded_si_bb2[s] == 0.0] = np.nan
        padded_si_bb2[s][padded_si_bb2[s] < 0] = 0.0
        padded_lin_bb2[s][padded_lin_bb2[s] == 0.0] = np.nan

    padded_lin_bb[padded_lin_bb == 0.0] = np.nan

    xticks = np.arange(0, N, dtype=np.int64)

    n_attacks_lin_bb = padded_lin_bb.shape[0]
    success_at_i_lin_wb = np.zeros(shape=N, dtype=float)
    success_at_i_lin_bb = np.zeros(shape=N, dtype=float)
    n_attacks_lin_bb2 = {}
    success_at_i_lin_bb2 = {}

    for s in strategies:
        n_attacks_lin_bb2[s] = padded_lin_bb2[s].shape[0]
        success_at_i_lin_bb2[s] = np.zeros(shape=N, dtype=float)

    for i in range(N-1):

        thiscol, nextcol = padded_lin_bb[:, i], padded_lin_bb[:, i + 1]
        next_step_inj = nextcol[~np.isnan(nextcol)].shape[0]
        this_step_inj = thiscol[~np.isnan(thiscol)].shape[0]
        success_at_i_lin_bb[i] = (this_step_inj - next_step_inj) / float(n_attacks_lin_bb)

        for s in strategies:
            thiscol, nextcol = padded_lin_bb2[s][:, i], padded_lin_bb2[s][:, i + 1]
            next_step_inj = nextcol[~np.isnan(nextcol)].shape[0]
            this_step_inj = thiscol[~np.isnan(thiscol)].shape[0]
            success_at_i_lin_bb2[s][i] = (this_step_inj - next_step_inj) / float(n_attacks_lin_bb2[s])

    cm_lin_wb = np.insert(np.cumsum(success_at_i_lin_wb), 0, 0)
    cm_lin_bb = np.insert(np.cumsum(success_at_i_lin_bb), 0, 0)
    cm_lin_bb2_cons = np.insert(np.cumsum(success_at_i_lin_bb2["conservative_0.25_ss"]), 0, 0)
    cm_lin_bb2_aggr = np.insert(np.cumsum(success_at_i_lin_bb2["aggressive_0.15_ss"]), 0, 0)

    # print(bb2_si_lin[strat], bb2_si_lin[strat] / np.sum(bb2_si_lin[strat]))
    plt.figure(figsize=(16, 9))

    t3 = np.array(n_atts_tot["iterative_step_1"])
    t4 = np.array(n_atts_tot["iterative_step_10"])

    success_at_i_1 = []
    success_at_i_10 = []

    for i in range(0, 1000):
        success_at_i_1.append(t3[t3<i].shape[0]/t3.shape[0])
        success_at_i_10.append(t4[t4<i].shape[0]/t4.shape[0])

    success_at_i_1 = np.array(success_at_i_1)
    success_at_i_10 = np.array(success_at_i_10)

    max_r_bb2_cons = bb2_lin_counter["conservative_0.25_ss"] / (bb_lin_counter)
    max_r_bb2_aggr = bb2_lin_counter["aggressive_0.15_ss"] / (bb_lin_counter)
    cm_lin_bb2_cons = cm_lin_bb2_cons * max_r_bb2_cons
    cm_lin_bb2_aggr = cm_lin_bb2_aggr * max_r_bb2_aggr
    #cm_lin_bb2_aggr = np.hstack((
    #    cm_lin_bb2_aggr[:5]*[1.35, 1.32, 1.29, 1.26, 1.23],
    #    cm_lin_bb2_aggr[5:10]*[1.2, 1.17, 1.14, 1.11, 1.08],
    #    cm_lin_bb2_aggr[10:15]*[1.05, 1.04, 1.03, 1.02, 1.01],
    #    cm_lin_bb2_aggr[15:]))





    plt.plot(xticks, success_at_i_10[:N], c="red", label="iterative",
             markersize=12, markeredgewidth=1, markeredgecolor="k",
             linewidth=2, linestyle='-', marker="s", markevery=3)
    plt.plot(xticks, cm_lin_bb2_cons[:N], c="pink", label="conservative",
             markersize=12, markeredgewidth=1, markeredgecolor="k",
             linewidth=2, linestyle='-', marker="o", markevery=3)
    plt.plot(xticks, cm_lin_bb2_aggr[:N], c="aqua", label="aggressive",
             markersize=12, markeredgewidth=1, markeredgecolor="k",
             linewidth=2, linestyle='-', marker="^", markevery=3)
    plt.ylim((-0.05, 0.85))
    plt.grid()
    plt.xlim((-1, 141))
    plt.yticks(fontsize=28)
    plt.xticks(fontsize=28)
    plt.ylabel("Ratio of successful attacks", fontsize=28)
    plt.xlabel("Number of total attempts", fontsize=28)
    plt.legend(loc="lower right", fontsize=28)
    plt.savefig("heuristic_comp1.pdf", bbox_inches="tight")

    plt.figure(figsize=(16, 9))

    colors = ["pink", "aqua"]
    markers = ["o", "^"]
    labels = ["conservative", "aggressive"]
    mx = n_attacks_lin_bb
    #for s in strategies:
    #    mx = max(padded_si_bb2[s].shape[0], mx)

    success_at_n_fails_iterative_1 = []
    success_at_n_fails_iterative_10 = []
    t1 = np.array(n_atts["iterative_step_1"])
    t2 = np.array(n_atts["iterative_step_10"])

    for i in range(0, 500):
        success_at_n_fails_iterative_1.append(t1[t1<i].shape[0]/t1.shape[0])
        success_at_n_fails_iterative_10.append(t2[t2<i].shape[0]/t2.shape[0])

    print(t2.shape[0])

    plt.plot(np.arange(0, 500), success_at_n_fails_iterative_10, c="red", label="iterative",
             markersize=12, markeredgewidth=1, markeredgecolor="k",
             linewidth=2, linestyle='-',
             marker="s", markevery=3
             )

    for k, s in enumerate(strategies):
        print("==='%s'===\nn success attacks: %d" % (s, padded_si_bb2[s].shape[0]))
        success_at_n_fails = []
        for i in range(0, 141):
            n_successes = 0
            for j in range(padded_si_bb2[s].shape[0]):
                _temp = padded_si_bb2[s][j]
                _temp[np.isnan(_temp)] = 2
                n_zeros = np.bincount(_temp.astype(int))[0]
                if n_zeros <= i:
                    n_successes += 1
            success_at_n_fails.append(n_successes)
        success_at_n_fails = np.array(success_at_n_fails)
        plt.plot(np.arange(0, 141), success_at_n_fails/mx, c=colors[k], label=labels[k],
                 markersize=12, markeredgewidth=1, markeredgecolor="k",
                 linewidth=2, linestyle='-', marker=markers[k], markevery=3)

    plt.ylim((-0.05, 0.85))
    plt.grid()
    plt.xlim((-1, 141))
    plt.yticks(fontsize=28)
    plt.xticks(fontsize=28)
    plt.ylabel("Ratio of successful attacks", fontsize=28)
    plt.xlabel("Number of failed attempts", fontsize=28)
    plt.legend(loc="lower right", fontsize=28)

    plt.savefig("heuristic_comp2.pdf", bbox_inches="tight")
