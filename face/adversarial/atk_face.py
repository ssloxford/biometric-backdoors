import argparse
import json
import os
import shutil
import sys
from os.path import join, isfile

import numpy as np
import tensorflow as tf


# my imports
from face import face_utils as fut
from face.adversarial import adv_utils as aut
from matchers.matchers import get_matchers_list


def load_negative_set(test_users, user, model, meta_dir, outlier_fraction, n=100, m=10):
    n_users = len(test_users)
    _other_users = np.random.choice(n_users, n, replace=False)
    _other_users = np.array(test_users, dtype=str)[_other_users]
    negative_set = []
    for j, v in enumerate(_other_users):
        if v == user:
            continue
        sys.stdout.write("\r%d/%d loaded" % (j+1, len(_other_users)))
        sys.stdout.flush()
        if not isfile(join(meta_dir, v, "embeddings_%s.npy" % model.class_name)):
            x_v = fut.load_data(meta_dir, v, model_name=model.class_name)
            y_v = model.predict(x_v)
            np.save(join(meta_dir, v, "embeddings_%s.npy" % model.class_name), y_v)
        else:
            y_v = np.load(join(meta_dir, v, "embeddings_%s.npy" % model.class_name))
        inliers_mask = fut.filter_inliers(y_v, outlier_fraction)
        y_v = y_v[inliers_mask]
        _chosen = np.random.choice(range(y_v.shape[0]), m, replace=False)
        negative_set.append(y_v[_chosen])
    sys.stdout.write("\n")
    negative_set = np.array(negative_set).reshape(-1, y_v.shape[-1])
    
    return negative_set


def check_args(args, data_path, victims, adversaries):
    assert len(set(victims).intersection(set(adversaries))) == 0
    assert args.adversary in adversaries
    for v in args.users.split(","):
        assert v in victims
    # assert args.user in victims
    index_folder = os.path.join(data_path, "results", args.source_folder)
    assert os.path.isfile(os.path.join(index_folder, args.adversary, "images_w_glasses_final.npy"))
    #p_changes_fp = os.path.join(index_folder, args.adversary, args.user, "positive_changes.csv")
    #n_changes_fp = os.path.join(index_folder, args.adversary, args.user, "negative_changes.csv")
    #assert os.path.isfile(p_changes_fp)
    #assert os.path.isfile(n_changes_fp)
    #assert aut.file_len(p_changes_fp) > 1
    #assert aut.file_len(n_changes_fp) > 1
    #assert os.path.isfile(join(index_folder, args.adversary, args.user, "run_args.json"))
    return True


def fix_input(images, s_model, t_model, target_height, target_width):
    img_temp = np.copy(images)
    img_temp = fut.undo_preprocessing(img_temp, s_model)
    img_temp = fut.resize_images(img_temp, target_height, target_width).astype(float)
    img_temp = fut.do_preprocessing(img_temp, t_model)
    return img_temp


def count_accepted(clf, embeddings, threshold):
    predicted = clf.predict(embeddings)
    accepted_mask = predicted > threshold
    n_accepted = embeddings[accepted_mask].shape[0]
    return n_accepted, accepted_mask


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="python atk_face.py -s facenet-vgg_10 -t facenet-vgg_10 -u n000040 -a n000009")
    parser.add_argument('-s', '--source_folder', action='store', required=True,
                        help="We get the adversarial stuff from this folder")
    parser.add_argument('-t', '--target_folder', action='store', required=True,
                        help="We get the target system from this folder")
    parser.add_argument('-u', '--users', action='store')
    parser.add_argument('-a', '--adversary', action='store', required=True)
    parser.add_argument('-oa', '--outliers_adversary', action='store', default=0.75, type=float)
    parser.add_argument('-ou', '--outliers_users', action='store', default=0.1, type=float)
    parser.add_argument('-re', '--reload_embeddings', action='store_true')
    parser.add_argument('-fr', '--force_redo', action='store_true')
    parser.add_argument('--minimum_for_inj', action='store', default=0.1, type=float)
    parser.add_argument('--max_inj_samples', action='store', default=10, type=int)
    parser.add_argument('-p', '--path_config', action='store', type=str, default=os.path.join("..", "config.json"))

    # np.random.seed(42)

    args = parser.parse_args(argv[1:])

    # configuration file for face
    conf_face = json.load(open(args.path_config, "r"))
    conf_source = json.load(open(join(conf_face["paths"]["data"], "results", args.source_folder, "ccc.json"), "r"))
    conf_target = json.load(open(join(conf_face["paths"]["data"], "results", args.target_folder, "ccc.json"), "r"))

    source_model = args.source_folder.split("_")[0]
    target_model = args.target_folder.split("_")[0]

    victims = conf_source["victims"]
    adversaries = conf_source["adversaries"]
    if args.users is None:
        args.users = ",".join(victims)
    # do a few checks for consistency of arguments
    check_args(args, conf_face["paths"]["data"], victims, adversaries)

    # define folders and stuff
    test_users_tar = list(conf_target["test_users"].keys())
    source_folder = join(conf_face["paths"]["data"], "results", args.source_folder)
    models_folder = join(conf_face["paths"]["data"], "models")

    np.set_printoptions(precision=8, suppress=True)

    # load image data
    meta_dir_source = fut.get_meta_folder(config=conf_face, model_name=source_model)
    meta_dir_source = os.path.join(conf_face["paths"]["data"], "dataset", meta_dir_source)

    meta_dir_target = fut.get_meta_folder(config=conf_face, model_name=target_model)
    meta_dir_target = os.path.join(conf_face["paths"]["data"], "dataset", meta_dir_target)

    # this is here for convenience, if called with -fr delete all embeddings and need restart
    if args.reload_embeddings:
        for v in test_users_tar:
            embeddings_file = join(meta_dir_target, v, "embeddings_%s.npy" % target_model)
            if os.path.exists(embeddings_file):
                os.remove(embeddings_file)
        print("[WARN] - Deleted embeddings for %s, run me again without the '-fr' arg" % target_model)
        exit()

    X_adv_tar_orig = fut.load_data(meta_dir_target, args.adversary, model_name=target_model)
    X_adv_src_orig = fut.load_data(meta_dir_source, args.adversary, model_name=source_model)

    users = args.users.split(",")

    # get things we need from source model first
    with tf.Graph().as_default():
        model_src = fut.model_names[source_model](models_folder=models_folder)
        y_adv_src_orig = model_src.predict(X_adv_src_orig)
        i_mask_adv_src = fut.filter_inliers(y_adv_src_orig, args.outliers_adversary, max_samples=50)
        X_adv_src_orig, y_adv_src_orig = X_adv_src_orig[i_mask_adv_src], y_adv_src_orig[i_mask_adv_src]

    # now we can start target model
    with tf.Graph().as_default():
        # load model target
        model_tar = fut.model_names[target_model](models_folder=models_folder)
        target_height, target_width, _ = model_tar.input_shape

        # compute user inliers to check for acceptance
        y_adv_tar_orig = model_tar.predict(X_adv_tar_orig)
        i_mask_adv_tar = fut.filter_inliers(y_adv_tar_orig, args.outliers_adversary)
        X_adv_tar_orig, y_adv_tar_orig = X_adv_tar_orig[i_mask_adv_tar], y_adv_tar_orig[i_mask_adv_tar]

        # load data to recreate the attack
        offsets = np.loadtxt(join(source_folder, args.adversary, "offsets.csv"), delimiter=",").astype(int)
        glasses_masks = np.load(join(source_folder, args.adversary, "glasses_masks.npy"))
        assert offsets.shape[0] == glasses_masks.shape[0] == X_adv_src_orig.shape[0] == y_adv_src_orig.shape[0]

        for user in users:
            # initialize output
            oof = join(source_folder, args.adversary, user, args.target_folder)

            if args.force_redo:
                print("[WARN] - deleting %s bc force_redo" % oof)
                shutil.rmtree(oof, ignore_errors=True)

            # check if we want to run this again
            if os.path.isfile(os.path.join(oof, "results.json")):
                print("[INFO] - %s -> %s - skip bc results.json exists already" % (args.adversary, user))
                continue
            print("[INFO] - %s -> %s" % (args.adversary, user))

            os.makedirs(oof, exist_ok=True)
            results = {}

            # load user data
            X_usr_tar = fut.load_data(meta_dir_target, user, model_name=target_model)
            run_args = json.load(open(join(source_folder, args.adversary, user, "run_args.json"), "r"))

            # load data for false accept rate
            y_others_tar = load_negative_set(
                test_users_tar, user, model_tar, meta_dir_target, args.outliers_users
            )

            # compute train and test masks for the user
            y_usr_tar = model_tar.predict(X_usr_tar)
            i_mask_usr_tar = fut.filter_inliers(y_usr_tar, args.outliers_users)
            X_usr_tar, y_usr_tar = X_usr_tar[i_mask_usr_tar], y_usr_tar[i_mask_usr_tar]
            n_samples_tar = conf_target["test_users"][user]["n_samples"]
            train_mask_tar = aut.mask_from_index(conf_target["test_users"][user]["chosen_train"], n_samples_tar)
            test_mask_tar = ~train_mask_tar
            target_mask_tar = aut.mask_from_index(conf_target["test_users"][user]["chosen_targets"], n_samples_tar)
            y_usr_train_tar = y_usr_tar[train_mask_tar]
            y_usr_test_tar = y_usr_tar[test_mask_tar]
            y_usr_target_tar = y_usr_tar[target_mask_tar].mean(axis=0)

            plus = np.loadtxt(join(source_folder, args.adversary, user, "positive_changes.csv"), dtype=int)
            minus = np.loadtxt(join(source_folder, args.adversary, user, "negative_changes.csv"), dtype=int)

            # loop over classifiers
            for ww in ["flat", "sigmoid"]:
                classifiers = get_matchers_list(ww)
                results[ww] = {}
                for clf in classifiers:
                    # initialize
                    clf_n = clf.get_name()
                    results[ww][clf_n] = {}
                    iar, far, frr = [], [], []
                    print("[INFO] - %s, %s, %s" % (args.adversary, user, clf_n))
                    trainset_tar = np.copy(y_usr_train_tar)

                    # load starting samples and final ones just for checking success
                    X_adv_g_src = np.load(join(source_folder, args.adversary, "images_w_glasses_final.npy"))
                    X_adv_g_f_src = np.load(join(source_folder, args.adversary, user, "images_w_glasses_final.npy"))

                    # fit classifier
                    clf.fit(trainset_tar)
                    clf_threshold_tar = conf_target["thr_%s" % clf_n]

                    m = 1

                    def shorter(imgs):
                        return fix_input(imgs, source_model, target_model, target_height, target_width)

                    y_adv_g_tar = model_tar.predict(shorter(X_adv_g_src))
                    y_adv_g_f_tar = model_tar.predict(shorter(X_adv_g_f_src))

                    n_adv_acc_g_tar, _ = count_accepted(clf, y_adv_g_tar, clf_threshold_tar*m)
                    n_adv_acc_g_f_tar, _ = count_accepted(clf, y_adv_g_f_tar, clf_threshold_tar*m)

                    # these three things we need to save
                    n_adv_acc_tar, _ = count_accepted(clf, y_adv_tar_orig, clf_threshold_tar)
                    n_usr_acc_tar, _ = count_accepted(clf, y_usr_test_tar, clf_threshold_tar)
                    n_others_acc_tar, _ = count_accepted(clf, y_others_tar, clf_threshold_tar)

                    iar.append(n_adv_acc_tar/y_adv_tar_orig.shape[0])
                    frr.append(n_usr_acc_tar/y_usr_test_tar.shape[0])
                    far.append(n_others_acc_tar/y_others_tar.shape[0])

                    # if no other outcome is registered we use '10' as the base
                    results[ww][clf_n]["outcome"] = 10

                    if n_adv_acc_tar/y_adv_tar_orig.shape[0] > .5:
                        results[ww][clf_n]["outcome"] = 1
                        print("[WARN] - skip because already %d/%d samples are accepted" % (n_adv_acc_tar, y_adv_tar_orig.shape[0]))
                        continue
                    print("[INFO] - # accepted plain %d/%d" % (n_adv_acc_tar, y_adv_tar_orig.shape[0]))
                    print("[INFO] - # accepted glasses start %d/%d" % (n_adv_acc_g_tar, y_adv_g_tar.shape[0]))
                    print("[INFO] - # accepted glasses final %d/%d" % (n_adv_acc_g_f_tar, y_adv_g_f_tar.shape[0]))

                    if n_adv_acc_g_f_tar/y_adv_g_tar.shape[0] < args.minimum_for_inj:
                        results[ww][clf_n]["outcome"] = 2
                        print("[INFO] dist test <-> target", aut.elem_wise_l2(y_usr_test_tar, y_usr_target_tar).mean())
                        print("[INFO] dist adv <-> target", aut.elem_wise_l2(y_adv_tar_orig, y_usr_target_tar).mean())
                        print("[INFO] dist g_adv_start <-> target", aut.elem_wise_l2(y_adv_g_tar, y_usr_target_tar).mean())
                        print("[INFO] dist g_adv_final <-> target", aut.elem_wise_l2(y_adv_g_f_tar, y_usr_target_tar).mean())
                        print("[WARN] - failed, not enough samples accepted (%d/%d at the end of optimization)" % (n_adv_acc_g_f_tar, y_adv_g_tar.shape[0]))
                        continue
                    injected_samples = []
                    injected_samples_perturbations = []
                    injected_samples_l2 = []
                    injected_samples_l1 = []
                    attempts = 0
                    try:
                        while attempts < args.max_inj_samples:
                            X_adv_g_src = np.load(join(source_folder, args.adversary, "images_w_glasses_final.npy"))
                            for i in range(plus.shape[0]):
                                X_adv_g_src = aut.gds_step_bobby(
                                    X_adv_src_orig, X_adv_g_src, plus[i], minus[i], offsets, run_args["theta"], run_args["clip_min"],
                                    run_args["clip_max"], glasses_masks
                                )
                                if i % run_args["smooth_frequency"] == 0:
                                    X_adv_g_src = aut._smooth_one_and_go(X_adv_g_src, glasses_masks.reshape(glasses_masks.shape[0], -1), run_args["gaussian_sigma"])
                                    X_adjusted_for_tar = shorter(X_adv_g_src)
                                    y_adv_g_tar = model_tar.predict(X_adjusted_for_tar)
                                    # l2d = aut.elem_wise_l2(y_adv_g, y_usr_target)
                                    adv_acc_mask_g_tar = clf.predict(y_adv_g_tar) > (clf_threshold_tar * m)
                                    n_adv_acc_g_tar = y_adv_g_tar[adv_acc_mask_g_tar].shape[0]
                                    #print("[INFO] - %d # accepted glasses %d/%d, (%f)" % (i, n_adv_acc_g, y_adv_g.shape[0], clf.predict(y_adv_g).mean()))
                                if n_adv_acc_g_tar/y_adv_g_tar.shape[0] >= args.minimum_for_inj:
                                    X_adjusted_for_tar = shorter(X_adv_g_src)
                                    # choose one among the accepted ones
                                    accepted_s_tar = y_adv_g_tar[adv_acc_mask_g_tar]
                                    index = np.random.choice(range(0, accepted_s_tar.shape[0]), 1)[0]
                                    trainset_tar = np.vstack((trainset_tar, accepted_s_tar[index]))
                                    injected_samples.append(X_adjusted_for_tar[adv_acc_mask_g_tar][index])
                                    injected_samples_perturbations.append(i)
                                    injected_samples_l2.append(aut.elem_wise_l2(y_adv_g_tar, y_usr_target_tar).tolist())
                                    injected_samples_l1.append(aut.elem_wise_l1(y_adv_g_tar, y_usr_target_tar).tolist())
                                    clf.fit(trainset_tar)

                                    # these three things we need to save
                                    n_adv_acc_tar, _ = count_accepted(clf, y_adv_tar_orig, clf_threshold_tar)
                                    n_usr_acc_tar, _ = count_accepted(clf, y_usr_test_tar, clf_threshold_tar)
                                    n_others_acc_tar, _ = count_accepted(clf, y_others_tar, clf_threshold_tar)

                                    iar.append(n_adv_acc_tar / y_adv_tar_orig.shape[0])
                                    frr.append(n_usr_acc_tar / y_usr_test_tar.shape[0])
                                    far.append(n_others_acc_tar / y_others_tar.shape[0])
                                    # print(iar[-1], frr[-1], far[-1])
                                    break
                            # check whether we reached the maximum number of perturbations (i == plus.shape[0])
                            # and the sample is not accepted, in that case we failed
                            # todo
                            attempts += 1
                            adv_acc_mask_tar = clf.predict(y_adv_tar_orig) > clf_threshold_tar
                            n_adv_acc_tar = y_adv_tar_orig[adv_acc_mask_tar].shape[0]
                            print("[INFO] - injected %d samples, %d/%d accepted" % (trainset_tar.shape[0] - y_usr_train_tar.shape[0], n_adv_acc_tar, y_adv_tar_orig.shape[0]))
                    except ValueError as e:
                        results[ww][clf_n]["outcome"] = 6
                        print("[WARNING] - attack failed, glasses masks mismatch size")
                        continue
                    results[ww][clf_n]["iar"] = iar
                    results[ww][clf_n]["frr"] = frr
                    results[ww][clf_n]["far"] = far
                    if n_adv_acc_tar/y_adv_tar_orig.shape[0] > 0.5:
                        results[ww][clf_n]["outcome"] = 0
                        results[ww][clf_n]["perturbations"] = injected_samples_perturbations
                        results[ww][clf_n]["l1"] = injected_samples_l1
                        results[ww][clf_n]["l2"] = injected_samples_l2
                        print("[INFO] - success")
                    else:
                        print("[INFO] - failed, not enough samples accepted after poisoning")
                        results[ww][clf_n]["outcome"] = 3
                    if len(injected_samples) > 0:
                        np.save(join(oof, "%s_images.npy" % clf_n), injected_samples)

            json.dump(results, open(join(oof, "results.json"), "w"))


if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt as e:
        print("Interrupted, %s" % str(e))

