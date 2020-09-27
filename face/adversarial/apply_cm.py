import argparse
import json
import os
import sys
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine

# my inputs
import adv_utils as aut
import face_utils as fut
from matchers.matchers import get_matchers_list


def check_args(args, data_path, victims, adversaries):
    assert len(set(victims).intersection(set(adversaries))) == 0
    assert args.adversary in adversaries
    # assert args.user in victims
    index_folder = os.path.join(data_path, "results", args.source_folder)
    assert os.path.isfile(os.path.join(index_folder, args.adversary, "images_w_glasses_final.npy"))
    return True



def get_angular_sims(train_data_o, update_samples):
    if update_samples.shape[0] < 2:
        return []
    train_data = np.copy(train_data_o)
    train_data = np.vstack((train_data, poisoning_emb[0]))
    delta_ys = [train_data[-1] - train_data.mean(axis=0)]
    cos_sims = []
    for i, sample in enumerate(update_samples[1:]):
        train_data = np.vstack((train_data, sample[np.newaxis, :]))
        delta_ys.append([train_data[-1] - train_data.mean(axis=0)])
        cos_sims.append(1 - cosine(delta_ys[-1], delta_ys[-2]))
        # print("Iter %d, tdata size %d, cos_sim %.4f" % (i, train_data.shape[0], cos_sims[-1]))
    similarities = list(cos_sims)
    return similarities



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Applies countermeasure")
    parser.add_argument('-s', '--source_folder', action='store', required=True,
                        help="We get the adversarial stuff from this folder")
    parser.add_argument('-t', '--target_folder', action='store', required=True,
                        help="We get the target system from this folder")
    parser.add_argument('-u', '--users', action='store')
    parser.add_argument('-a', '--adversary', action='store', required=True)
    parser.add_argument('-v', '--visualize', action='store_true', default=False)
    parser.add_argument('-fr', '--force_redo', action='store_true')
    parser.add_argument('-ou', '--outliers_users', action='store', default=0.1, type=float)
    parser.add_argument('-oa', '--outliers_adversary', action='store', default=0.75, type=float)
    parser.add_argument('-th', '--threshold', action="store", type=float, help="images/countermeasure_table.py script will give you this")
    parser.add_argument('-p', '--path_config', action='store', type=str, default=os.path.join("..", "config.json"))

    args = parser.parse_args(sys.argv[1:])

    # configuration file for face
    conf_face = json.load(open(args.path_config, "r"))
    conf_source = json.load(open(join(conf_face["paths"]["data"], "results", args.source_folder, "ccc.json"), "r"))

    model_name = args.source_folder.split("_")[0]

    victims = conf_source["victims"]
    adversaries = conf_source["adversaries"]
    test_users = list(conf_source["test_users"].keys())

    source_folder = join(conf_face["paths"]["data"], "results", args.source_folder)
    # define folders and users
    models_folder = join(conf_face["paths"]["data"], "models")
    # do a few checks for consistency of arguments
    check_args(args, conf_face["paths"]["data"], victims, adversaries)

    meta_dir = fut.get_meta_folder(config=conf_face, model_name=model_name)
    meta_dir = os.path.join(conf_face["paths"]["data"], "dataset", meta_dir)
    np.set_printoptions(precision=8, suppress=True)

    X_adv = fut.load_data(meta_dir, args.adversary, model_name=model_name)
    model = fut.model_names[model_name](models_folder=models_folder)
    y_adv = model.predict(X_adv)
    i_mask_adv = fut.filter_inliers(y_adv, args.outliers_adversary, max_samples=50)
    X_adv, y_adv = X_adv[i_mask_adv], y_adv[i_mask_adv]

    with model.persistent_sess as sess:
        for user in args.users.split(","):
            # let's only consider the same-model for now
            r_folder = join(source_folder, args.adversary, user, args.target_folder)
            # initialize output
            results = {}
            # check if we want to run this again
            #if os.path.isfile(os.path.join(r_folder, "counter.json")):
            #    print("[INFO] - Counterm %s -> %s - skip bc counter.json exists already" % (args.adversary, user))
            #    continue

            print("[INFO] - Counterm %s -> %s" % (args.adversary, user))
            #results = {}

            # run arguments for this user
            run_args = json.load(open(join(source_folder, args.adversary, user, "run_args.json"), "r"))
            # load and check data
            X_usr = fut.load_data(meta_dir, user, model_name=model_name)
            img_paths_usr_test = fut.get_image_paths(meta_dir, user)
            assert X_usr.shape[0] == len(img_paths_usr_test)

            # load and filters outliers out
            y_usr = model.predict(X_usr)
            i_mask_usr = fut.filter_inliers(y_usr, args.outliers_users)
            X_usr = X_usr[i_mask_usr]
            y_usr = y_usr[i_mask_usr]

            # now load and filter by the split information
            n_samples = conf_source["test_users"][user]["n_samples"]
            train_mask = aut.mask_from_index(conf_source["test_users"][user]["chosen_train"], n_samples)
            test_mask = ~train_mask
            target_mask = aut.mask_from_index(conf_source["test_users"][user]["chosen_targets"], n_samples)
            y_usr_train = y_usr[train_mask]
            y_usr_test = y_usr[test_mask]
            y_usr_target = y_usr[target_mask].mean(axis=0)

            oof = join(source_folder, args.adversary, user, args.source_folder)
            results = json.load(open(os.path.join(oof, "results.json"), "r"))

            new_results = {}

            for ww in ["flat", "sigmoid"]:
                classifiers = get_matchers_list(ww)
                for clf in classifiers:
                    # initialize
                    clf_n = clf.get_name()
                    if clf_n not in ["cnt_flat"]:
                        continue

                    new_results[clf_n] = {}
                    new_results[clf_n]["iar_pre"] = []
                    new_results[clf_n]["iar_post"] = []

                    if results[ww][clf_n]["outcome"] == 0:
                        # then we can actually check things
                        iar = results[ww][clf_n]["iar"]
                        iar_with_cm = [iar[0]]
                        poisoning_images = np.load(join(r_folder, "%s_images.npy" % clf_n))
                        poisoning_emb = model.predict(poisoning_images)

                        # train clf
                        trainset = np.array(y_usr_train)
                        clf.fit(trainset)

                        ang_sim_p = get_angular_sims(trainset, poisoning_emb)

                        # inject first guy
                        #trainset = np.vstack((trainset, poisoning_emb[0]))
                        iar_with_cm.append(iar[1])

                        assert len(ang_sim_p) == poisoning_emb.shape[0]-1

                        for jj in range(1, poisoning_emb.shape[0]):
                            cur_ps = poisoning_emb[jj]
                            ang_sim = ang_sim_p[jj-1]  # ignore the first on
                            
                            if ang_sim > args.threshold:

                                # attack detected
                                # now the injected sample is discarded
                                # and the previous sample is discarded as they are too similar
                                # trainset = trainset[:-1, :]
                                iar_with_cm.append(iar_with_cm[-2])
                                # print("detected", ang_sim, iar_with_cm)
                                break

                            else:
                                #trainset = np.vstack((trainset, poisoning_emb[jj+1]))
                                #clf.fit(trainset)
                                iar_with_cm.append(iar[jj+1])
                                # print("undetected", ang_sim, iar_with_cm)

                        while len(iar_with_cm) < 11:
                            iar_with_cm.append(iar_with_cm[-1])
                        
                        # print(iar_with_cm)
                        new_results[clf_n]["iar_post"] = iar_with_cm
                        new_results[clf_n]["iar_pre"] = iar
            fpath_to_save = os.path.join(r_folder, "applied_cm.json")
            json.dump(new_results, open(fpath_to_save, "w"), indent=2)










