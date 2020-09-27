import argparse
import json
import os
from os.path import join
import numpy as np
from face import face_utils as fut
import plot_utils as put
import pyperclip
import socket
import datetime


def get_success(conf, source_folders, adversaries, victims):

    output = {}
    success_bound = .5  # 50% of accepted template

    for model in source_folders:
        output[model] = {}
        for w in ["flat", "sigmoid"]:
            for c in ["cnt", "mxm", "lsvm"]:
                output[model]["%s_%s" % (c, w)] = {}
                output[model]["%s_%s" % (c, w)]["successes"] = 0
                output[model]["%s_%s" % (c, w)]["failures"] = 0
                output[model]["%s_%s" % (c, w)]["one"] = []
                output[model]["%s_%s" % (c, w)]["three"] = []
                output[model]["%s_%s" % (c, w)]["ten"] = []


    for model in source_folders:
        folder_path = join(conf["paths"]["data"], "results", model)
        for a in adversaries:
            for v in victims:
                results_filepath = join(folder_path, a, v, model, "results.json")
                if not os.path.isfile(results_filepath):
                    continue
                results = json.load(open(results_filepath, "r"))
                for w in ["flat", "sigmoid"]:
                    for c in ["cnt", "mxm", "lsvm"]:
                        if results[w]["%s_%s" % (c, w)]["outcome"] == 0:
                            output[model]["%s_%s" % (c, w)]["successes"] += 1
                            output[model]["%s_%s" % (c, w)]["one"].append(int(results[w]["%s_%s" % (c, w)]["iar"][1]>success_bound))
                            output[model]["%s_%s" % (c, w)]["three"].append(int(results[w]["%s_%s" % (c, w)]["iar"][3]>success_bound))
                            output[model]["%s_%s" % (c, w)]["ten"].append(int(results[w]["%s_%s" % (c, w)]["iar"][10]>success_bound))
                        if results[w]["%s_%s" % (c, w)]["outcome"] == 2 or results[w]["%s_%s" % (c, w)]["outcome"] == 3:
                            output[model]["%s_%s" % (c, w)]["failures"] += 1
    return output

txt_filename = "big-result.tex"

label = "tab:big_result"

caption = """
Attack success rates for each considered model and classifier.
Success is defined as >50\% IAR after $i$ samples injections (either one, three or ten).
Each figure is calculated on the (same) 1000 randomly chosen pairs of attacker-victim.
"""


if __name__ == "__main__":

    np.random.seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_folders', action='store', default="vggresnet-vgg2_10,facenet-vgg2_10,vgg16-lfw_10")
    parser.add_argument('-p', '--path_config', action='store', type=str, default=os.path.join("..", "face", "config.json"))
    parser.add_argument("-d", "--destination", type=str)


    args = parser.parse_args()

    source_folders = args.source_folders.split(",")

    # configuration file for face
    conf_face = json.load(open(args.path_config, "r"))
    conf_results = json.load(open(join(conf_face["paths"]["data"], "results", source_folders[0], "ccc.json"), "r"))

    victims = conf_results["victims"]
    adversaries = conf_results["adversaries"]
    test_users = list(conf_results["test_users"].keys())

    successes = get_success(conf_face, source_folders, adversaries, victims)

    def inj_line():
        s = "\\textit{Model}&\\textit{1\\textsuperscript{st}} & \\textit{3\\textsuperscript{rd}} & \\textit{10\\textsuperscript{th}} & \\textit{1\\textsuperscript{st}} & \\textit{3\\textsuperscript{rd}} & \\textit{10\\textsuperscript{th}} & \\textit{1\\textsuperscript{st}} & \\textit{3\\textsuperscript{rd}} & \\textit{10\\textsuperscript{th}} & \\textit{1\\textsuperscript{st}} & \\textit{3\\textsuperscript{rd}} & \\textit{10\\textsuperscript{th}} & \\textit{1\\textsuperscript{st}} & \\textit{3\\textsuperscript{rd}} & \\textit{10\\textsuperscript{th}} & \\textit{1\\textsuperscript{st}} & \\textit{3\\textsuperscript{rd}} & \\textit{10\\textsuperscript{th}} \\\\"
        return s

    print(successes)
    output = "\\begin{table*}[t]\n"
    output += "\centering\n"
    output += "\\begin{tabular}{cccc|ccc|ccc|ccc|ccc|ccc}\n"
    output += "\\toprule\n"
    output += "& \multicolumn{6}{c}{\\texttt{centroid}} & \multicolumn{6}{c}{\\texttt{maximum}} &\multicolumn{6}{c}{\\texttt{SVM}}\\\\"
    output += "\cmidrule(l){2-7}\cmidrule(l){8-13}\cmidrule(l){14-19}"
    # output += "\\textit{Model} & \\textit{flat} & \\textit{sigm} & \\textit{flat} & \\textit{sigm} & \\textit{flat} & \\textit{sigm} \\\\\n"
    output += "& \multicolumn{3}{c}{\\textit{flat}} & \multicolumn{3}{c}{\\textit{sigmoid}} & \multicolumn{3}{c}{\\textit{flat}} & \multicolumn{3}{c}{\\textit{sigmoid}}& \multicolumn{3}{c}{\\textit{flat}} & \multicolumn{3}{c}{\\textit{sigmoid}} \\\\"
    output += inj_line()

    for model_name in sorted(successes.keys()):
        this = successes[model_name]
        output += "\midrule\n"
        output += "%s " % put.model_to_names[model_name.split("_")[0]]

        total_runs = {}
        clfs = ["cnt_flat", "cnt_sigmoid", "mxm_flat", "mxm_sigmoid", "lsvm_flat", "lsvm_sigmoid"]
        for clf in clfs:
            total_runs[clf] = this[clf]["failures"] + this[clf]["successes"]

        for clf in clfs:
            for step in ["one", "three", "ten"]:
                output += " & {0:.0f}\%".format(sum(this[clf][step]) *100 / (total_runs[clf]+1))

        output += "\\\\\n"

        #output += "{0:.0f}\% & {1:.0f}\% & {2:.0f}\% & {3:.0f}\% & {4:.0f}\% & {5:.0f}\%\\\\\n".format(
        #    this["cnt_flat"]["successes"]*100 / (this["cnt_flat"]["failures"] + this["cnt_flat"]["successes"]+1),
        #    this["cnt_sigmoid"]["successes"]*100 / (this["cnt_sigmoid"]["failures"] + this["cnt_sigmoid"]["successes"] + 1),
        #    this["mxm_flat"]["successes"]*100 / (this["mxm_flat"]["failures"] + this["mxm_flat"]["successes"]+1),
        #    this["mxm_sigmoid"]["successes"]*100 / (this["mxm_sigmoid"]["failures"] + this["mxm_sigmoid"]["successes"] + 1),
        #    this["lsvm_flat"]["successes"]*100 / (this["lsvm_flat"]["failures"] + this["lsvm_flat"]["successes"]+1),
        #    this["lsvm_sigmoid"]["successes"]*100 / (this["lsvm_sigmoid"]["failures"] + this["lsvm_sigmoid"]["successes"]+1),
        #)
    output += "\\bottomrule\n"
    output += "\end{tabular}\n"
    output += "\caption{%s}\n" % caption
    output += "\label{%s}\n" % label
    output += "% Generated on {0:s} at {1:s}\n".format(socket.gethostname(), str(datetime.datetime.now()))
    output += "\end{table*}\n"
    print(output)
    pyperclip.copy(output)

    if args.destination is not None:
        print("[INFO] - creating %s" % os.path.join(args.destination, txt_filename))
        with open(os.path.join(args.destination, txt_filename), "w") as outfile:
            outfile.write(output)
        print("[INFO] - done")
