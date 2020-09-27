import json
from os.path import join
import pyperclip
import plot_utils as put
import socket
import datetime
import argparse
import shutil
import os

subf = sorted(["facenet-vgg2", "vgg16-lfw", "vggresnet-vgg2"])
ts_sizes = list(map(str, [10, ]))
clf_names = ["cnt", "mxm", "lsvm"]
weights = ["flat", "sigmoid"]

txt_filename = "table-eer.tex"

label = "tab:eer"

caption = """
Performance of the face-recognition models in terms of EER.
The classifiers thresholds are set at the EER (where false accept and false reject rates are equal) computed on the
development set.
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--destination", type=str)
    args = parser.parse_args()

    config_face = json.load(open("../face/config.json", "r"))
    base = config_face["paths"]["data"]
    results_folder_base = join(base, "results")

    output = "\\begin{table}[t]\n"
    output += "\centering\n"
    output += "\\begin{tabular}{ccc|cc|cc}\n"
    output += "\\toprule\n"
    output += "& \multicolumn{2}{c}{\\texttt{centroid}} & \multicolumn{2}{c}{\\texttt{maximum}} &\multicolumn{2}{c}{\\texttt{SVM}}\\\\\n"
    output += "\\textit{Model} & \\textit{\\textsc{eer}} & \\textit{thr.} & \\textit{\\textsc{eer}} & \\textit{thr.} & \\textit{\\textsc{eer}} & \\textit{thr.} \\\\\n"

    for f in subf:
        line = "\midrule\n"
        line = line + "%s" % put.model_to_names[f]
        for size in ts_sizes:
            folder_name = f + "_" + size
            conf_file = json.load(open(join(results_folder_base, folder_name, "ccc.json"), "r"))

            for clf in clf_names:
                for w in weights:
                    this_eer = conf_file["eer_%s_%s" % (clf, w)]*100
                    this_auc = conf_file["auc_%s_%s" % (clf, w)]
                    this_thr = -conf_file["thr_%s_%s" % (clf, w)]
#
                    if this_auc > 0.99:
                        this_auc = ">.99"

                    this_thr = "{0:.3f}".format(this_thr)

                    this_eer = "{0:.1f}\%".format(this_eer)

                    # if this_eer < 1:
                    #     this_eer = "<1\%"
                    # else:
                    #     this_eer = "{0:.0f}\%".format(this_eer)

                    line = line + " & %s & %s" % (this_eer, this_thr)
                    break
        line = line + "\\\\"
        output += line

    output += "\\bottomrule\n"
    output += "\end{tabular}\n"
    output += "\caption{%s}\n" % caption
    output += "\label{%s}\n" % label
    output += "% Generated on {0:s} at {1:s}\n".format(socket.gethostname(), str(datetime.datetime.now()))
    output += "\end{table}\n"
    print(output)
    pyperclip.copy(output)

    if args.destination is not None:
        print("[INFO] - creating %s" % os.path.join(args.destination, txt_filename))
        with open(os.path.join(args.destination, txt_filename), "w") as outfile:
            outfile.write(output)
        print("[INFO] - done")
