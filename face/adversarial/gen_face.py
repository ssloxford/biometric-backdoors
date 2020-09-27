import argparse
import json
import multiprocessing
import os

import numpy as np
import scipy.misc
import tensorflow as tf


# my imports
from face import face_utils as fut
from face.adversarial import adv_utils as aut
from face.adversarial import tf_utils as tut


def check_args(args, victims, adversaries):
    for v in args.users.split(","):
        assert v in victims
    assert args.adversary in adversaries
    assert args.ncores <= multiprocessing.cpu_count()
    return True


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="python gen_face.py -f facenet-vgg_10 -u n000009 -a n000040")
    parser.add_argument('-c', '--ncores', action='store', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('-i', '--maxiters', action='store', type=int)
    parser.add_argument('-t', '--theta_multiplier', action='store', type=float)
    parser.add_argument('-f', '--indexfolder', action='store', required=True)
    parser.add_argument('-u', '--users', action='store')
    parser.add_argument('-n', '--n_pixels', action='store', type=int)
    parser.add_argument('-g', '--gaussian_sigma', action='store', type=float)
    parser.add_argument('-l', '--n_images', action='store', type=int)
    parser.add_argument('-a', '--adversary', action='store', required=True)
    parser.add_argument('-ou', '--outliers_users', action="store", type=float)
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('-fr', '--force_redo', action="store_true")
    parser.add_argument('--convergence_after', action="store", type=int)
    parser.add_argument('--smooth_frequency', action='store', type=int)
    parser.add_argument('-p', '--path_config', action='store', type=str, default=os.path.join("..", "config.json"))

    script_name = argv[0]

    args = parser.parse_args(argv[1:])

    model_name = args.indexfolder.split("_")[0]
    assert model_name in fut.model_names

    # configuration file for face
    conf_face = json.load(open(args.path_config, "r"))
    conf_res_fp = os.path.join(conf_face["paths"]["data"], "results", args.indexfolder, "ccc.json")
    conf_res = json.load(open(conf_res_fp, "r"))

    # define folders and users
    results_folder = os.path.join(conf_face["paths"]["data"], "results", args.indexfolder)
    models_folder = os.path.join(conf_face["paths"]["data"], "models")
    victims = sorted(list(conf_res["victims"]))
    adversaries = sorted(list(conf_res["adversaries"]))

    if args.users is None:
        args.users = ",".join(victims)

    args = aut.load_args(model_name, conf_face, script_name, args)
    print("[INFO] - ", args)

    # do a few checks for consistency of arguments
    check_args(args, victims, adversaries)

    meta_dir = fut.get_meta_folder(config=conf_face, model_name=model_name)
    meta_dir = os.path.join(conf_face["paths"]["data"], "dataset", meta_dir)

    # these are the victims we want to attack
    users = args.users.split(",")

    # this line loads the model
    model = fut.model_names[model_name](models_folder=models_folder)
    model.convert_to_classifier()

    # load adversary information
    glasses_masks = np.load(os.path.join(results_folder, args.adversary, "glasses_masks.npy"))
    print("[INFO] - glasses shape, %s" % str(glasses_masks.shape))
    n_pix_glasses = glasses_masks[0].astype(int).sum()
    n_pix_input = np.product(glasses_masks[0].shape)
    print("[INFO] - # pixels in glasses: %d (%.2f%%)" % (n_pix_glasses, n_pix_glasses*100/n_pix_input))

    offsets = np.loadtxt(os.path.join(results_folder, args.adversary, "offsets.csv"), delimiter=",", dtype=int)
    X_adv_orig = np.load(os.path.join(results_folder, args.adversary, "images_w_glasses_final.npy"))
    assert X_adv_orig.shape[0] == glasses_masks.shape[0] == offsets.shape[0]

    # find an appropriate theta value
    clip_min = X_adv_orig.min()
    clip_max = X_adv_orig.max()
    theta = np.abs(clip_max - clip_min) / 255.0 * args.theta_multiplier
    print("[INFO] - Min, max, theta: %f, %f, %f" % (clip_min, clip_max, theta))

    # masks
    glasses_masks = glasses_masks.reshape(glasses_masks.shape[0], -1)
    n_images = args.n_images if args.n_images != -1 else X_adv_orig.shape[0]

    # initialize tensors
    theta_ph = tf.placeholder(tf.float32, name="args_theta")
    batch_size_ph = tf.placeholder(tf.int32, name='batch_size')

    glasses_masks_t = tf.convert_to_tensor(glasses_masks[:n_images], dtype=tf.bool)
    offsets = tf.convert_to_tensor(offsets[:n_images], dtype=tf.int32)
    x_adv, b1, b2, loss, var = tut.fgsm_m(
        model, theta_ph, clip_min, clip_max, glasses_masks_t, offsets, args.n_pixels, i_shape=model.input_shape
    )
    for user in users:
        if os.path.isfile(os.path.join(results_folder, args.adversary, user, "run_args.json")) and not args.force_redo:
            print("[INFO] - %s -> %s - skip bc run_args.json exists already" % (args.adversary, user))
            continue
        print("[INFO] - %s -> %s" % (args.adversary, user))
        X_adv = np.copy(X_adv_orig)

        os.makedirs(os.path.join(results_folder, args.adversary, user), exist_ok=True)
        # now we need to compute the embedding target for the optimization
        # in this case, this is the known user sample
        X_usr = fut.load_data(meta_dir, user, model_name=model_name)
        y_usr = model.predict(X_usr)
        i_mask_usr = fut.filter_inliers(y_usr, args.outliers_users)
        X_usr, y_usr = X_usr[i_mask_usr], y_usr[i_mask_usr]

        # load user data and target
        n_samples = conf_res["test_users"][user]["n_samples"]
        target_mask = aut.mask_from_index(conf_res["test_users"][user]["chosen_targets"], n_samples)
        y_usr = y_usr[target_mask]
        centroid_usr = y_usr.mean(axis=0)

        # find target
        y_target = np.tile(centroid_usr, (X_adv.shape[0], 1))[:n_images]

        # start logging stuff
        pos_changes = []
        neg_changes = []
        distance_at_i = []

        emb_w_glasses = model.predict(X_adv[:n_images])
        distance_at_i.append(aut.elem_wise_l2(emb_w_glasses, centroid_usr))
        print("[INFO] - %d/%d" % (0, args.maxiters), distance_at_i[-1].mean(), distance_at_i[-1].std())

        for iteration in range(args.maxiters):
            # tt = max(0.01, theta/(int(iteration/40) + 1))
            # print(tt)
            feed_dict = {
                model.face_input: X_adv[:n_images],
                model.pt: False,
                model.target_embedding_input: y_target[:n_images],
                batch_size_ph: 10,
                theta_ph: theta if iteration < args.maxiters / 2.0 else theta / 2.0
            }
            X_adv, pos, neg, _, _var = model.persistent_sess.run([x_adv, b1, b2, loss, var], feed_dict=feed_dict)
            # print(_, _var)

            if iteration % args.smooth_frequency == 0 or iteration == args.maxiters - 1:
                X_adv = aut._smooth_one_and_go(X_adv, glasses_masks, args.gaussian_sigma)

            pos_changes.append(pos)
            neg_changes.append(neg)

            emb_w_glasses = model.predict(X_adv)

            distance_at_i.append(aut.elem_wise_l2(emb_w_glasses, centroid_usr))
            if iteration % 10 == 0:
                print("[INFO] - %d/%d" % (iteration + 1, args.maxiters), distance_at_i[-1].mean(), distance_at_i[-1].std())

            # print(distance_at_i[-1].mean() - distance_at_i[max(0, iteration-100)].mean(), 0.01)
            if iteration > args.convergence_after:
                if iteration % args.smooth_frequency == 0:
                    if distance_at_i[max(0, iteration - args.convergence_after)].mean() - distance_at_i[-1].mean() < 0.0:
                        print("[INFO] - stop optimization bc distance didn't change for iter")
                        break

        pos_changes = np.array(pos_changes, dtype=int)
        neg_changes = np.array(neg_changes, dtype=int)
        distance_at_i = np.array(distance_at_i)

        np.savetxt(os.path.join(results_folder, args.adversary, user, "positive_changes.csv"), pos_changes, fmt="%d")
        np.savetxt(os.path.join(results_folder, args.adversary, user, "negative_changes.csv"), neg_changes, fmt="%d")
        np.savetxt(os.path.join(results_folder, args.adversary, user, "distance_at_i.csv"), distance_at_i)
        np.save(os.path.join(results_folder, args.adversary, user, "images_w_glasses_final.npy"), X_adv)
        run_args = vars(args)
        run_args["clip_min"] = float(clip_min)
        run_args["clip_max"] = float(clip_max)
        run_args["theta"] = float(theta)
        json.dump(
            run_args,
            open(os.path.join(results_folder, args.adversary, user, "run_args.json"), "w")
        )

        if args.visualize:
            for x in X_adv:
                print(x.shape)
                scipy.misc.imshow(x)



    #with tf.Graph().as_default():#
    #    with model.persistent_sess as sess:



if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt as e:
        print("Interrupted, %s" % str(e))

