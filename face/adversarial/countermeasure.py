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


def load_pose_stuff(image_paths, ):
    user_meta_folder = os.path.sep.join(image_paths[0].split(os.path.sep)[:-1])
    pose_info = json.load(open(os.path.join(user_meta_folder, "vision.json"), "r"))
    poses = []
    for i in range(len(image_paths)):
        img_filename = image_paths[i].split(os.path.sep)[-1]
        pose = -180
        try:
            # print(img_filename, pose_info[img_filename]["face_annotations"][0]["pan_angle"])
            pose = pose_info[img_filename]["face_annotations"][0]["pan_angle"]
        except (IndexError, KeyError) as e:
            pass
            # print("Skip %s" % img_filename)
        poses.append(pose)
        # print(img_filename, pose)
    return poses


def load_glasses_stuff(image_paths, ):
    user_meta_folder = os.path.sep.join(image_paths[0].split(os.path.sep)[:-1])
    glasses_info = json.load(open(os.path.join(user_meta_folder, "vision.json"), "r"))
    glasses_l = []
    for i in range(len(image_paths)):
        img_filename = image_paths[i].split(os.path.sep)[-1]
        glasses = False
        try:
            # print(img_filename, pose_info[img_filename]["face_annotations"][0]["pan_angle"])
            la = glasses_info[img_filename]["label_annotations"]
            for item in la:
                if item["description"] == "Sunglasses":
                    glasses = True
                    break
        except (IndexError, KeyError) as e:
            pass
            # print("Skip %s" % img_filename)
        glasses_l.append(glasses)
        # print(img_filename, pose)
    return glasses_l


def load_beard_stuff(image_paths, ):
    user_meta_folder = os.path.sep.join(image_paths[0].split(os.path.sep)[:-1])
    beard_info = json.load(open(os.path.join(user_meta_folder, "vision.json"), "r"))
    beard_l = []
    for i in range(len(image_paths)):
        img_filename = image_paths[i].split(os.path.sep)[-1]
        beard = False
        try:
            # print(img_filename, pose_info[img_filename]["face_annotations"][0]["pan_angle"])
            la = beard_info[img_filename]["label_annotations"]
            for item in la:
                if item["description"] == "Facial hair":
                    beard = True
                    break
        except (KeyError, IndexError) as e:
            pass
            # print("Skip %s" % img_filename)
        beard_l.append(beard)
        # print(img_filename, pose)
    return beard_l


def load_age_stuff(image_paths, ):
    user_meta_folder = os.path.sep.join(image_paths[0].split(os.path.sep)[:-1])
    age_info = json.load(open(os.path.join(user_meta_folder, "max-age-estimate.json"), "r"))
    ages = []
    for i in range(len(image_paths)):
        img_filename = image_paths[i].split(os.path.sep)[-1]
        age = -1
        try:
            # print(img_filename, pose_info[img_filename]["face_annotations"][0]["pan_angle"])
            age = age_info[img_filename]["predictions"][0]["age_estimation"]
        except (IndexError, KeyError) as e:
            pass
            # print("Skip %s" % img_filename)
        ages.append(age)
        # print(img_filename, age)
    return ages


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
        description="Compute attacks detected by our angular similarity-based countermeasure")
    parser.add_argument('-s', '--source_folder', action='store', required=True,
                        help="We get the adversarial stuff from this folder")
    parser.add_argument('-t', '--target_folder', action='store', required=True,
                        help="We get the target system from this folder")
    parser.add_argument('-u', '--users', action='store')
    parser.add_argument('-a', '--adversary', action='store', required=True)
    parser.add_argument('-v', '--visualize', action='store_true', default=False)
    parser.add_argument('-fr', '--force_redo', action='store_true')
    parser.add_argument('-ou', '--outliers_users', action='store', default=0.1, type=float)
    parser.add_argument('-p', '--path_config', action='store', type=str, default=os.path.join("..", "config.json"))

    args = parser.parse_args(sys.argv[1:])

    # configuration file for face
    conf_face = json.load(open(args.path_config, "r"))
    conf_source = json.load(open(join(conf_face["paths"]["data"], "results", args.source_folder, "ccc.json"), "r"))
    conf_target = json.load(open(join(conf_face["paths"]["data"], "results", args.target_folder, "ccc.json"), "r"))

    source_model = args.source_folder.split("_")[0]
    target_model = args.target_folder.split("_")[0]

    victims = conf_source["victims"]
    adversaries = conf_source["adversaries"]
    test_users_src = list(conf_source["test_users"].keys())
    test_users_tar = list(conf_target["test_users"].keys())

    source_folder = join(conf_face["paths"]["data"], "results", args.source_folder)
    # define folders and users
    models_folder = join(conf_face["paths"]["data"], "models")
    # do a few checks for consistency of arguments
    check_args(args, conf_face["paths"]["data"], victims, adversaries)

    meta_dir_tar = fut.get_meta_folder(config=conf_face, model_name=target_model)
    meta_dir_tar = os.path.join(conf_face["paths"]["data"], "dataset", meta_dir_tar)
    np.set_printoptions(precision=8, suppress=True)

    X_adv_tar = fut.load_data(meta_dir_tar, args.adversary, model_name=target_model)
    model_tar = fut.model_names[target_model](models_folder=models_folder)

    with model_tar.persistent_sess as sess:

        for user in args.users.split(","):
            # let's only consider the same-model for now
            r_folder = join(source_folder, args.adversary, user, args.target_folder)


            if args.force_redo:
                try:
                    print("[WARN] - Counterm, delete %s bc force_redo" % os.path.join(r_folder, "counter.json"))
                    os.remove(os.path.join(r_folder, "counter.json"))
                except Exception as e:
                    pass
            # initialize output
            results = {}
            # check if we want to run this again
            if os.path.isfile(os.path.join(r_folder, "counter.json")):
                print("[INFO] - Counterm %s -> %s - skip bc counter.json exists already" % (args.adversary, user))
                continue

            if not os.path.isfile(os.path.join(meta_dir_tar, user, "vision.json")) or not os.path.isfile(os.path.join(meta_dir_tar, user, "max-age-estimate.json")):
                print("[WARN] - Counterm %s -> %s - skip bc either vision.json or age-estimate.json don't exist" % (args.adversary, user))
                continue
            print("[INFO] - Counterm %s -> %s" % (args.adversary, user))
            results = {}

            # run arguments for this user
            run_args = json.load(open(join(source_folder, args.adversary, user, "run_args.json"), "r"))
            # load and check data
            X_usr_tar = fut.load_data(meta_dir_tar, user, model_name=target_model)
            img_paths_usr_test = fut.get_image_paths(meta_dir_tar, user)
            assert X_usr_tar.shape[0] == len(img_paths_usr_test)

            # load and filters outliers out
            y_usr_tar = model_tar.predict(X_usr_tar)
            i_mask_usr_tar = fut.filter_inliers(y_usr_tar, args.outliers_users)
            X_usr_tar = X_usr_tar[i_mask_usr_tar]
            y_usr_tar = y_usr_tar[i_mask_usr_tar]

            # now load and filter by the split information
            n_samples_tar = conf_target["test_users"][user]["n_samples"]
            train_mask_tar = aut.mask_from_index(conf_target["test_users"][user]["chosen_train"], n_samples_tar)
            test_mask_tar = ~train_mask_tar
            target_mask_tar = aut.mask_from_index(conf_target["test_users"][user]["chosen_targets"], n_samples_tar)
            y_usr_train_tar = y_usr_tar[train_mask_tar]
            y_usr_test_tar = y_usr_tar[test_mask_tar]
            y_usr_target_tar = y_usr_tar[target_mask_tar].mean(axis=0)

            # get image_filepaths for each factor
            img_paths_usr_test = np.array(img_paths_usr_test)[i_mask_usr_tar][test_mask_tar]
            poses = np.array(load_pose_stuff(img_paths_usr_test))
            glasses = np.array(load_glasses_stuff(img_paths_usr_test))
            beards = np.array(load_beard_stuff(img_paths_usr_test))
            ages = np.array(load_age_stuff(img_paths_usr_test))

            # get masks on the image filepaths for each factor
            pose_mask = np.bitwise_or(poses >= 25, poses <= -25)
            pose_mask = np.bitwise_and(pose_mask, poses != -180)
            glasses_mask = np.array(glasses, dtype=bool)
            beards_mask = np.array(beards, dtype=bool)
            age_min = ages.min()
            age_max = ages.max()
            age_spans = np.linspace(age_min, age_max, 5).astype(int)
            age_masks = []
            for i in range(len(age_spans) - 1):
                _m1 = np.bitwise_and(ages >= age_spans[i], ages < age_spans[i + 1])
                _m2 = np.bitwise_and(_m1, ages != -1)
                age_masks.append(_m2)

            # filter by the above masks
            y_usr_test_pose = y_usr_test_tar[pose_mask]
            y_usr_test_glasses = y_usr_test_tar[glasses_mask]
            y_usr_test_beards = y_usr_test_tar[beards_mask]
            y_usr_test_age = [y_usr_test_tar[age_mask] for age_mask in age_masks]

            img_fp_pose = img_paths_usr_test[pose_mask]
            img_fp_age = [img_paths_usr_test[age_mask] for age_mask in age_masks]
            img_fp_glasses = img_paths_usr_test[glasses_mask]
            img_fp_beards = img_paths_usr_test[beards_mask]

            img_fp_pose_fn = list(map(lambda x: x.split(os.path.sep)[-1], img_fp_pose.tolist()))
            img_fp_glasses_fn = list(map(lambda x: x.split(os.path.sep)[-1], img_fp_glasses.tolist()))
            img_fp_beards_fn = list(map(lambda x: x.split(os.path.sep)[-1], img_fp_beards.tolist()))
            img_fp_random_fn = list(map(lambda x: x.split(os.path.sep)[-1], img_paths_usr_test.tolist()))

            results["random_images"] = img_fp_random_fn
            results["pose_images"] = img_fp_pose_fn
            results["glasses_images"] = img_fp_glasses_fn
            results["beards_images"] = img_fp_beards_fn
            results["age_images"] = {}

            for i in range(len(age_spans)-1):
                results["age_images"][i] = {}
                results["age_images"][i]["from"] = int(age_spans[i])
                results["age_images"][i]["to"] = int(age_spans[i+1])
                images_fn = list(map(lambda x: x.split(os.path.sep)[-1], img_fp_age[i].tolist()))
                results["age_images"][i]["images"] = images_fn

            # print(results)

            for ww in ["flat", "sigmoid"]:
                classifiers = get_matchers_list(ww)
                for clf in classifiers:
                    # initialize
                    clf_n = clf.get_name()
                    results[clf_n] = {}
                    # print(join(r_folder, "%s_images.npy" % clf_n))
                    if not os.path.isfile(join(r_folder, "%s_images.npy" % clf_n)):
                        results[clf_n]["cos_dist_age"] = []
                        results[clf_n]["cos_dist_poison"] = []
                        results[clf_n]["cos_dist_pose"] = []
                        results[clf_n]["cos_dist_glasses"] = []
                        results[clf_n]["cos_dist_beards"] = []
                        results[clf_n]["cos_dist_random"] = []
                        continue
                    poisoning_images = np.load(join(r_folder, "%s_images.npy" % clf_n))
                    poisoning_emb = model_tar.predict(poisoning_images)

                    poisoning_upd = get_angular_sims(y_usr_train_tar, poisoning_emb)
                    random_upd = get_angular_sims(y_usr_train_tar, y_usr_test_tar)
                    pose_upd = get_angular_sims(y_usr_train_tar, y_usr_test_pose)
                    glasses_upd = get_angular_sims(y_usr_train_tar, y_usr_test_glasses)
                    beards_upd = get_angular_sims(y_usr_train_tar, y_usr_test_beards)

                    age_upd_tmp = [get_angular_sims(y_usr_train_tar, y_age) for y_age in y_usr_test_age]
                    age_upd = []
                    for u in age_upd_tmp:
                        age_upd.extend(u)

                    results[clf_n]["cos_dist_random"] = random_upd
                    results[clf_n]["cos_dist_age"] = list(map(float, age_upd))
                    results[clf_n]["cos_dist_poison"] = poisoning_upd
                    results[clf_n]["cos_dist_pose"] = pose_upd
                    results[clf_n]["cos_dist_glasses"] = glasses_upd
                    results[clf_n]["cos_dist_beards"] = beards_upd

                    if args.visualize:
                        plt.hist(random_upd, histtype="step", bins=np.linspace(-1, 1, 50), color="red")
                        plt.hist(pose_upd, histtype="step",bins=np.linspace(-1, 1, 50), color="green")
                        plt.hist(age_upd, histtype="step",bins=np.linspace(-1, 1, 50), color="cyan")
                        plt.hist(glasses_upd, histtype="step",bins=np.linspace(-1, 1, 50), color="yellow")
                        plt.hist(beards_upd, histtype="step",bins=np.linspace(-1, 1, 50), color="black")
                        plt.hist(poisoning_upd, histtype="step",bins=np.linspace(-1, 1, 50), color="blue")
                        plt.show()

            json.dump(results, open(os.path.join(r_folder, "counter.json"), "w"), indent=2)















