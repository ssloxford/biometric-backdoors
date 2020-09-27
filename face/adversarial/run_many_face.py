import argparse
import json
import os
import subprocess
import sys


def check_exists(fold, subfolders, files, heur=""):
    for f in files:
        for sf in subfolders:
            if not os.path.isfile(os.path.join(fold, sf, heur, f)):
                return False
    return True


description = """
This script can run the whole evaluation after preprocessing.

This takes a -w <ARG> argument which will run all the scripts specified in the <ARG>, where 
s - will run start_face.py
g - will run gen_face.py
a - will run atk_face.py
c - will run countermeasure.py
p - will run apply_cm.py
so `python.py -s <SF> -t <TF> -w sgacp` would run all the necessary scripts for those folders.

when -s and -t point to different folders the scripts are run across different models (transferability)
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-s', '--source_folder', action='store', required=True)
    parser.add_argument('-t', '--target_folder', action='store', required=True)
    parser.add_argument('-w', '--which_scripts', action='store', required=True)
    parser.add_argument('-r', '--force_redo', action='store_true')
    parser.add_argument('-p', '--path_config', action='store', type=str, default="../config.json")

    args = parser.parse_args(sys.argv[1:])

    # configuration file for face
    conf_face = json.load(open(args.path_config, "r"))
    conf_res = json.load(open(os.path.join(conf_face["paths"]["data"], "results", args.source_folder, "ccc.json"), "r"))

    # define folders and users
    results_folder = os.path.join(conf_face["paths"]["data"], "results")
    dataset_folder = os.path.join(conf_face["paths"]["data"], "dataset")
    models_folder = os.path.join(conf_face["paths"]["data"], "models")

    source_folder = os.path.join(results_folder, args.source_folder)

    users = conf_res["victims"]
    adversaries = conf_res["adversaries"]

    selected_pairs = json.load(open(os.path.join(results_folder, "selected_pairs.json"), "r"))

    source_model = args.source_folder.split("_")[0]
    target_model = args.target_folder.split("_")[0]

    print(source_model, "->", target_model)

    adversaries = list(sorted(set([x for x, y in selected_pairs])))
    users = list(sorted(set([y for x, y in selected_pairs])))

    print("Adversaries: %s" % ",".join(adversaries))
    print("Users: %s" % ",".join(users))

    a = input("Does that look good?[y/n]")
    if a != "y":
        print("Interrupted")
        exit()

    scripts = list(args.which_scripts)

    done_so_far = 0

    cm_thresholds = {
        "facenet-vgg2_10": 0.40976,
        "vggresnet-vgg2_10": 0.41365,
        "vgg16-lfw_10": 0.48204
    }

    for _, a in enumerate(adversaries):

        # adv_seed = abs(hash(a)) % (10 ** 8) * model_seed % 2**32 - 1
        # users = shuffle(users, random_state=adv_seed)[:50]
        users = list(sorted([y for x, y in selected_pairs if x == a]))

        start_script = "python start_face.py -f %(source_folder)s -a %(adversary)s" % {
            "adversary": a,
            "source_folder": args.source_folder
        }

        if "s" in scripts:
            if not args.force_redo and os.path.isfile(os.path.join(results_folder, args.source_folder, a, "images_w_glasses_final.npy")):
                print("[INFO] - Start, %s, SKIP, images_w_glasses_final.npy exists already" % a)
            else:
                print("[INFO] - Start, %s" % a)
                subprocess.call(start_script.split(" "))

        gen_script = "python gen_face.py -f %(source_folder)s -a %(adversary)s -u%(users)s %(force_redo)s" % {
            "source_folder": args.source_folder,
            "adversary": a,
            "users": ",".join(users),
            "force_redo": " " if not args.force_redo else "-fr"
        }
        if "g" in scripts:
            # do a smart check to see if we run this script already
            if os.path.isfile(os.path.join(source_folder, a, users[-1], "run_args.json")) and not args.force_redo:
                print("[WARN] - Generate %s skip because last user %s was executed already" % (a, users[-1]))
            else:
                print("[INFO] - Generate, %s" % a)
                subprocess.call(gen_script.rstrip(" ").split(" "))

        atk_script = "python atk_face.py -s %(source_folder)s -t %(target_folder)s -a %(adversary)s -u %(users)s %(force_redo)s" % {
            "source_folder": args.source_folder,
            "target_folder": args.target_folder,
            "adversary": a,
            "users": ",".join(users),
            "force_redo": " " if not args.force_redo else "-fr"
        }

        if "a" in scripts:
            # do a smart check to see if we run this script already
            last_user_f = os.path.join(source_folder, a, users[-1], args.target_folder)
            if os.path.isfile(os.path.join(last_user_f, "results.json")) and not args.force_redo:
                print("[WARN] - Attack %s skipping because last user %s results.json exists already" % (a, users[-1]))
            else:
                print("[INFO] - Attack, %s, %s -> %s" % (a, args.source_folder, args.target_folder))
                subprocess.call(atk_script.rstrip(" ").split(" "))

        cm_script = "python countermeasure.py -s %(source_folder)s -t %(target_folder)s -a %(adversary)s -u %(users)s %(force_redo)s" % {
            "source_folder": args.source_folder,
            "target_folder": args.target_folder,
            "adversary": a,
            "users": ",".join(users),
            "force_redo": " " if not args.force_redo else "-fr"
        }
        if "c" in scripts:
            # do a smart check to see if we run this script already
            last_user_f = os.path.join(source_folder, a, users[-1], args.target_folder)
            if os.path.isfile(os.path.join(last_user_f, "counter.json")) and not args.force_redo:
                print("[WARN] - Countermeasure %s skipping because last user %s counter.json exists already" % (a, users[-1]))
            else:
                print("[INFO] - Countermeasure, %s, %s -> %s" % (a, args.source_folder, args.target_folder))
                subprocess.call(cm_script.rstrip(" ").split(" "))

        app_cm_script = "python apply_cm.py -s %(source_folder)s -t %(target_folder)s -a %(adversary)s -u %(users)s -th %(cm_thresh)s %(force_redo)s" % {
            "source_folder": args.source_folder,
            "target_folder": args.target_folder,
            "adversary": a,
            "users": ",".join(users),
            "cm_thresh": str(cm_thresholds[args.source_folder]),
            "force_redo": " " if not args.force_redo else "-fr"
        }

        if "p" in scripts:
            # do a smart check to see if we run this script already
            last_user_f = os.path.join(source_folder, a, users[-1], args.target_folder)
            if os.path.isfile(os.path.join(last_user_f, "applied_cm.json")) and not args.force_redo:
                print("[WARN] - Apply countermeasure %s skipping because last user %s applied_cm.json exists already" % (a, users[-1]))
            else:
                print("[INFO] - Apply countermeasure, %s, %s -> %s" % (a, args.source_folder, args.target_folder))
                subprocess.call(app_cm_script.rstrip(" ").split(" "))

        done_so_far += len(users)
        print("[INFO] - Adv %s (%d/1000), victims: %s" % (a, done_so_far, ",".join(users)))
