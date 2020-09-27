import argparse
import json
import multiprocessing
import os

import numpy as np
import pandas as pd
import scipy.misc
import tensorflow as tf

# my imports
from face import face_utils as fut
from face.adversarial import adv_utils as aut
from face.adversarial import tf_utils as tut


def check_args(args, data_path, users):
    assert args.adversary in users
    assert args.ncores <= multiprocessing.cpu_count()
    assert os.path.isfile(os.path.join(data_path, "results", args.index_folder, "ccc.json"))
    return True


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="python start_face.py -f facenet-vgg_10 -a n000009")
    parser.add_argument('-c', '--ncores', action='store', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('-i', '--maxiters', action='store', type=int)
    parser.add_argument('-r', '--n_random_glasses', action='store', type=int)
    parser.add_argument('-f', '--index_folder', action='store', required=True)
    parser.add_argument('-a', '--adversary', action='store', required=True)
    parser.add_argument('-n', '--n_pixels', action='store', type=int)
    parser.add_argument('-g', '--gaussian_sigma', action='store', type=float)
    parser.add_argument('-t', '--theta_multiplier', action='store', type=float)
    parser.add_argument('-l', '--n_images', action='store', type=int)
    parser.add_argument('-oa', '--outliers_adversary', action='store', type=float)
    parser.add_argument('--convergence_after', action="store", type=int)
    parser.add_argument('--smooth_frequency', action='store', type=int)
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('-p', '--path_config', action='store', type=str, default=os.path.join("..", "config.json"))

    script_name = argv[0]

    args = parser.parse_args(argv[1:])

    # configuration file for face
    conf_face = json.load(open(args.path_config, "r"))

    # define folders and users
    results_folder = os.path.join(conf_face["paths"]["data"], "results", args.index_folder)
    conf_res = json.load(open(os.path.join(results_folder, "ccc.json"), "r"))
    models_folder = os.path.join(conf_face["paths"]["data"], "models")
    users = sorted(conf_res["adversaries"])

    model_name = args.index_folder.split("_")[0]
    meta_dir = fut.get_meta_folder(config=conf_face, model_name=model_name)
    meta_dir = os.path.join(conf_face["paths"]["data"], "dataset", meta_dir)
    args = aut.load_args(model_name, conf_face, script_name, args)
    print(args)

    # do a few checks for consistency of arguments
    check_args(args, conf_face["paths"]["data"], users)

    with tf.Graph().as_default():
        model = fut.model_names[model_name](models_folder=models_folder)
        with model.persistent_sess as sess:
            # load model and create folders
            adv_folder = os.path.join(results_folder, args.adversary)
            os.makedirs(adv_folder, exist_ok=True)

            # find a sample in feature space whose projection in embedding space is sufficiently close to y_centroid
            # we find this by optimizing the glasses so that they change the embedding location of the sample
            # close to the centroid, for all samples.

            # landmarks contains 10 values, [x0, y0, x1, y1, ... x4, y4]
            # in order, they are:
            # p0 = left part of left eye
            # p1 = right part of left eye
            # p2 = right part of right eye
            # p3 = left part of right eye
            # p4 = nose
            X = fut.load_data(meta_dir, args.adversary, model_name)
            landmarks = pd.read_csv(os.path.join(meta_dir, args.adversary, "landmarks.csv"), index_col=0).values
            assert X.shape[0] == landmarks.shape[0]

            y = model.predict(X)
            i_mask = fut.filter_inliers(y, args.outliers_adversary, max_samples=50)
            X, y, landmarks = X[i_mask], y[i_mask], landmarks[i_mask]
             
            clip_min, clip_max = X.min(), X.max()
            theta = np.abs(clip_max - clip_min)/255.0 * args.theta_multiplier
            print("[INFO] - # samples", X.shape[0], clip_min, clip_max, theta)

            # now we need to compute the embedding target for the optimization
            # in this case, this is the user centroid (mean)
            centroid_adv = y.mean(axis=0)
            
            glasses_path = os.path.join(conf_face["paths"]["repository"], "face", "adversarial", "glasses.bmp")
            glasses_masks, offsets = aut.get_all_glasses_masks(landmarks, glasses_path, model.input_shape)
            offsets = offsets[:, [0, 2]]

            l2_dist_best = 100
            glasses_best = []
            # iterates over a few random generated glasses colors
            for j in range(args.n_random_glasses):
                print("[INFO] - %d-th Random glasses" % (j+1))
                g_base = aut.init_random_glasses(glasses_masks[0], clip_min, clip_max, index_of_blue_channel=model.input_mode.find("B"))
                # print(g_base.min(),g_base.max())
                color_only = g_base[glasses_masks[0]].flatten()
                wearing_glasses = aut._local_apply_stuff(X, color_only, glasses_masks, model.input_shape)
                #scipy.misc.imshow(wearing_glasses[0])
                #scipy.misc.imshow(wearing_glasses[0])
                glasses_embeddings = model.predict(wearing_glasses)
                l2_dist = aut.elem_wise_l2(glasses_embeddings, centroid_adv)
                glasses_score = np.array(l2_dist).mean() + np.array(l2_dist).std()*5
                if glasses_score < l2_dist_best:
                    print("[INFO] - Found best pair of glasses:", glasses_score)
                    l2_dist_best = glasses_score
                    glasses_best = np.array(color_only)

            # save initial glasses
            # re-apply the best ones
            images_w_glasses = aut._local_apply_stuff(X, glasses_best, glasses_masks, model.input_shape)
            

            np.save(os.path.join(results_folder, args.adversary, "glasses_color.npy"), glasses_best)
            np.save(os.path.join(results_folder, args.adversary, "glasses_masks.npy"), glasses_masks)
            np.savetxt(os.path.join(results_folder, args.adversary, "offsets.csv"), offsets, delimiter=",", fmt="%d")

            # after here the optimization starts
            glasses_masks = glasses_masks.reshape(glasses_masks.shape[0], -1)

            n_images = X.shape[0] if args.n_images == -1 else args.n_images
            
            y_target = np.tile(centroid_adv, (X.shape[0], 1))[:n_images]
            glasses_masks_t = tf.convert_to_tensor(glasses_masks[:n_images], dtype=tf.bool)
            offsets_t = tf.convert_to_tensor(offsets[:n_images], dtype=tf.int32)
            # print("###############")
            model.convert_to_classifier()

            x_adv, b1, b2, _, _ = tut.fgsm_m(
                model, theta, clip_min, clip_max, glasses_masks_t, offsets_t, args.n_pixels, i_shape=model.input_shape
            )

            # start logging stuff
            positive_changes = []
            negative_changes = []
            distance_at_i = []

            for iteration in range(args.maxiters):
                feed_dict = {
                    model.face_input: images_w_glasses[:n_images],
                    model.pt: False,
                    model.target_embedding_input: y_target[:n_images]
                }
                images_w_glasses, pos, neg, t11, t22 = sess.run([x_adv, b1, b2, model.softmax_output, model.distance], feed_dict=feed_dict)
                # print(t11, t22)

                if iteration % args.smooth_frequency == 0 or iteration == args.maxiters - 1:
                    images_w_glasses = aut._smooth_one_and_go(images_w_glasses, glasses_masks, args.gaussian_sigma)
                
                positive_changes.append(pos)
                negative_changes.append(neg)

                emb_w_glasses = model.predict(images_w_glasses)

                distance_at_i.append(aut.elem_wise_l2(emb_w_glasses, centroid_adv))
                if iteration % 10 == 0:
                    print("[INFO] - %d/%d" % (iteration+1, args.maxiters), distance_at_i[-1].mean(), distance_at_i[-1].std())

                if iteration>args.convergence_after:
                    if iteration % args.smooth_frequency == 0:
                        index = max(0, iteration-args.convergence_after)
                        if distance_at_i[max(0, iteration-args.convergence_after)].mean() - distance_at_i[-1].mean() < 0.0:
                            print("[WARN] - stop optimization bc distance didn't change for iter")
                            break

            positive_changes = np.array(positive_changes, dtype=int)
            negative_changes = np.array(negative_changes, dtype=int)
            distance_at_i = np.array(distance_at_i)

            np.savetxt(os.path.join(results_folder, args.adversary, "positive_changes.csv"), positive_changes, fmt="%d")
            np.savetxt(os.path.join(results_folder, args.adversary, "negative_changes.csv"), negative_changes, fmt="%d")
            np.savetxt(os.path.join(results_folder, args.adversary, "distance_at_i.csv"), distance_at_i)
            np.save(os.path.join(results_folder, args.adversary, "images_w_glasses_final.npy"), images_w_glasses)
            json.dump(vars(args), open(os.path.join(results_folder, args.adversary, "dict.json"), "w"))
            if args.visualize:
                for x in images_w_glasses:
                    print(x.shape)
                    scipy.misc.imshow(x)


if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt as e:
        print("Interrupted, %s" % str(e))
