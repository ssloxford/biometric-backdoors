import sys
import os
import json
from sklearn.utils import shuffle
import itertools
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path_config', action='store', type=str, default="../config.json")
    args = parser.parse_args(sys.argv[1:])
    conf_face = json.load(open(args.path_config, "r"))
    conf_res = json.load(open(os.path.join(conf_face["paths"]["data"], "results", "facenet-vgg2_10", "ccc.json"), "r"))

    results_folder = os.path.join(conf_face["paths"]["data"], "results")

    users = conf_res["victims"]
    adversaries = conf_res["adversaries"]

    all_pairs = list(itertools.product(adversaries, users))
    selected_pairs = shuffle(all_pairs, random_state=42)[:1000]

    selected_pairs = sorted(selected_pairs, key=lambda tup: (tup[0], tup[1]))

    o_file = os.path.join(results_folder, "selected_pairs.json")

    if os.path.isfile(o_file):
        a = input("selected_pairs.json exists already in \n%s\ndo you want to overwrite it?[y/n]" % o_file)
        if a == "y":
            json.dump(selected_pairs, open(o_file, "w"), indent=2)
            print("[WARN] - overwritten selected_pairs.json")
        else:
            print("[INFO] - done nothing")
    else:
        json.dump(selected_pairs, open(o_file, "w"), indent=2)
        print("[INFO] - Written selected_pairs.json")

    




