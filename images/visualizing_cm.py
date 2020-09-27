from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
import argparse
import os
import json
import numpy as np
import time
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from face import face_utils as fut
from face.adversarial import adv_utils as aut
from os.path import join
import scipy.misc
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import plot_utils as put


fontsize = 36
msize = 200
malpha = 0.9
figsize=(16, 9)
annot_offset = 0.25



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


colors = [
        "b",
        "g",
        "r",
        "y",
        "cyan",
        "purple",
        "orange",
        "brown",
        "black",
        "pink"
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-s', '--source_folder', action='store', required=True, choices=["vggresnet-vgg2_10", "facenet-vgg2_10", "vgg16-lfw_10"])
    # parser.add_argument('-du', '--n_dev_users', action='store', default=10)
    # parser.add_argument('-da', '--n_dev_attackers', action='store', default=10)
    parser.add_argument('-p', '--path_config', action='store', type=str, default=os.path.join("..", "face", "config.json"))

    args = parser.parse_args()
    conf_face = json.load(open(args.path_config, "r"))
    source_folder = "facenet-vgg2_10"
    r_folder = os.path.join(conf_face["paths"]["data"], "results", source_folder)
    conf_results = json.load(open(os.path.join(r_folder, "ccc.json"), "r"))
    victims = conf_results["victims"]
    adversaries = conf_results["adversaries"]
    selected_pairs = json.load(open(join(conf_face["paths"]["data"], "results", "selected_pairs.json"), "r"))
    adversaries = list(sorted(set([x for x, y in selected_pairs])))
    users = list(sorted(set([y for x, y in selected_pairs])))

    dev_pairs = [[x, y] for x, y in selected_pairs]

    models_folder = os.path.join(conf_face["paths"]["data"], "models")
    model = fut.model_names["facenet-vgg2"](models_folder=models_folder)

    meta_dir = os.path.join(conf_face["paths"]["data"], "dataset", "meta-160-36")
    all = []
    labels = []

    for v in ["n000040"]:
        f = os.path.join(meta_dir, v, "embeddings_facenet-vgg2.npy")
        # print(f)
        if os.path.isfile(f):
            l = np.load(f)
            for i in l:
                all.append(i)
                labels.append(v)
            # all.append(np.load(f))
    emb_base, labels_base = shuffle(np.array(all), labels)
    emb_base = emb_base[:1]
    labels_base = labels_base[:1]

    meta_dir = fut.get_meta_folder(config=conf_face, model_name="facenet-vgg2")
    meta_dir = os.path.join(conf_face["paths"]["data"], "dataset", meta_dir)

    pca_50 = PCA(n_components=50)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

    indexx = 10

    dev_pairs = [["n000775", "n001242"]]

    for adv, user in dev_pairs:

        #if os.path.isfile("./plots/%s_%s.png" % (adv, user)):
        #    continue

        print(adv, user)
        X_adv_orig = fut.load_data(meta_dir, adv, model_name="facenet-vgg2")
        y_adv = model.predict(X_adv_orig)
        i_mask_allll = fut.filter_inliers(y_adv, .1, max_samples=50)
        y_adv_all = y_adv[i_mask_allll]
        i_mask_adv = fut.filter_inliers(y_adv, .75, max_samples=50)
        X_adv_orig, y_adv = X_adv_orig[i_mask_adv], y_adv[i_mask_adv]

        X_usr = fut.load_data(meta_dir, user, model_name="facenet-vgg2")
        y_usr = model.predict(X_usr)
        i_mask_usr = fut.filter_inliers(y_usr, .1)
        X_usr, y_usr = X_usr[i_mask_usr], y_usr[i_mask_usr]

        img_fpaths_usr = fut.get_image_paths(meta_dir, user)
        img_fpaths_usr = np.array(img_fpaths_usr)
        img_fpaths_usr = img_fpaths_usr[i_mask_usr]
        assert X_usr.shape[0] == y_usr.shape[0] == img_fpaths_usr.shape[0]
        try:
            pose_usr = load_pose_stuff(img_fpaths_usr)
            glasses_usr = load_glasses_stuff(img_fpaths_usr)
            beards_usr = load_beard_stuff(img_fpaths_usr)
        except Exception as e:
            print("Error in json, skip")
            continue

        def get_colors(inp, colormap, vmin=None, vmax=None):
            norm = plt.Normalize(vmin, vmax)
            return colormap(norm(inp))

        X_adv_g = np.load(join(r_folder, adv, "images_w_glasses_final.npy"))
        y_adv_g_start = model.predict(X_adv_g)

        plus = np.loadtxt(join(r_folder, adv, user, "positive_changes.csv"), dtype=int)
        minus = np.loadtxt(join(r_folder, adv, user, "negative_changes.csv"), dtype=int)
        run_args = json.load(open(join(r_folder, adv, user, "run_args.json"), "r"))
        offsets = np.loadtxt(join(r_folder, adv, "offsets.csv"), delimiter=",").astype(int)
        glasses_masks = np.load(join(r_folder, adv, "glasses_masks.npy"))
        assert offsets.shape[0] == glasses_masks.shape[0] == X_adv_orig.shape[0]

        first_100_steps = [y_adv_g_start[indexx]]
        try:
            for i in range(plus.shape[0]):
                X_adv_g = aut.gds_step_bobby(
                    X_adv_orig, X_adv_g, plus[i], minus[i], offsets, run_args["theta"]/4, run_args["clip_min"],
                    run_args["clip_max"], glasses_masks
                )

                #if i % run_args["smooth_frequency"] == 0:
                #    X_adv_g = aut._smooth_one_and_go(X_adv_g, glasses_masks.reshape(glasses_masks.shape[0], -1), run_args["gaussian_sigma"])
                if i in [int(1.75**i)-1 for i in range(12)] or i == plus.shape[0]:
                    # print(i)
                    y_adv_g = model.predict(X_adv_g)
                    first_100_steps.append(y_adv_g[indexx])
                    #scipy.misc.imshow(X_adv_g[0])
        except Exception as e:
            print("Error, continue")
            continue
        first_100_steps = np.array(first_100_steps)

        # now we stack the poisoning with the base samples for tsne
        poses_stacked = np.hstack((
            np.ones(shape=emb_base.shape[0])*100,
            np.ones(shape=y_adv_all.shape[0])*100,
            pose_usr
        ))
        glasses_stacked = np.hstack((
            np.zeros(shape=emb_base.shape[0]).astype(bool),
            np.zeros(shape=y_adv_all.shape[0]).astype(bool),
            glasses_usr
        ))
        beards_stacked = np.hstack((
            np.zeros(shape=emb_base.shape[0]).astype(bool),
            np.zeros(shape=y_adv_all.shape[0]).astype(bool),
            beards_usr
        ))

        pca_input = np.vstack((emb_base, y_adv_all, y_usr))
        # embedding_scaler = StandardScaler()
        # embedding_scaler.fit(pca_input)
        # pca_input = embedding_scaler.transform(pca_input)
        pca_labels = labels_base + ["adv" for j in range(y_adv_all.shape[0])] + ["vic" for j in range(y_usr.shape[0])]
        pca_50.fit(pca_input)
        pca_output = pca_50.transform(pca_input)
        print('Explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

        poses_stacked = np.hstack((
            poses_stacked, np.ones(shape=first_100_steps.shape[0])*100
        ))
        glasses_stacked = np.hstack((glasses_stacked, np.zeros(shape=first_100_steps.shape[0]).astype(bool)))
        beards_stacked = np.hstack((beards_stacked, np.zeros(shape=first_100_steps.shape[0]).astype(bool)))

        tsne_input = np.vstack((pca_output, pca_50.transform(first_100_steps)))
        tsne_labels = pca_labels + ["poison" for i in range(first_100_steps.shape[0])]
        assert tsne_input.shape[0] == len(tsne_labels) == poses_stacked.shape[0]

        rndperm = np.random.permutation(len(tsne_labels))
        time_start = time.time()
        tsne_results = tsne.fit_transform(tsne_input[rndperm])
        tsne_labels = np.array(tsne_labels)[rndperm]
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        consider = ["adv", "vic", "poison"]
        # consider = ["n000129", "n000736"]
        fig = plt.figure(figsize=figsize)
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.add_subplot(111)

        adversary_points = tsne_results[np.array(tsne_labels) == "adv"]
        ax.scatter(adversary_points[:, 0], adversary_points[:, 1], s=msize, marker="v", c="blue", alpha=malpha, label="adversary")

        poisoning_points = tsne_results[np.array(tsne_labels) == "poison"]
        # ax.scatter(poisoning_points[:, 0], poisoning_points[:, 1], s=msize, marker="^", c="red", alpha=malpha, label="poisoning sample")

        victim_mask = np.array(tsne_labels) == "vic"

        poses_victim = poses_stacked[rndperm][victim_mask]
        poses_victim[poses_victim==-180] = poses_victim[poses_victim>-180].min()
        pose_colors = get_colors(pose_usr, plt.cm.Greens)

        glasses_victim = glasses_stacked[rndperm][victim_mask]
        # glasses_colors = ["green" if x else "limegreen" for x in glasses_victim]

        beards_victim = beards_stacked[rndperm][victim_mask]
        # beards_colors = ["orange" if x else "yellow" for x in beards_victim]

        victim_points = tsne_results[victim_mask]

        vpoints_to_plot = victim_points[np.bitwise_and(~beards_victim, ~glasses_victim)]
        ax.scatter(vpoints_to_plot[:, 0], vpoints_to_plot[:, 1], s=msize, marker="o", c="green", alpha=malpha, label="victim")

        # now draw beards and glasses in different color
        glasses_p = victim_points[glasses_victim]
        beards_p = victim_points[beards_victim]

        ax.scatter(beards_p[:, 0], beards_p[:, 1], s=msize, marker="s", c="brown", alpha=malpha, label="victim (with beard)")

        victim_centroid = victim_points.mean(axis=0)
        adv_centroid = adversary_points.mean(axis=0)

        going_down = int((victim_centroid[1] - adv_centroid[1]) > 0)*2-1
        going_left = int((victim_centroid[0] - adv_centroid[0]) > 0)*2-1

        i_p_x = np.linspace(adv_centroid[0], victim_centroid[0], 30)
        i_p_y = np.linspace(adv_centroid[1], victim_centroid[1], 30)
        mask = np.array([False, False, True, True, False, True, False, False, True, False,
                         False, False, True, False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False, False, False, False])
        # plot poisoning points
        ax.scatter(i_p_x[mask], i_p_y[mask], s=msize * 1.2, marker="^", c="red", alpha=malpha, label="poisoning sample")

        # arrow 1
        #ax.arrow(interm_points_x[12], interm_points_y[12], interm_points_x[8], interm_points_y[8],
        #         head_width=1, head_length=1.0, fc="k", ec="k", alpha=0.95)
        arrow_1_annotate_coord = (
            (i_p_x[18] + i_p_x[12]) / 2 + annot_offset * going_left,
            (i_p_y[18] + i_p_y[12]) / 2 - annot_offset * going_down
        )

        arrow_2_annotate_coord = (
            (i_p_x[12] + i_p_x[8]) / 2 + annot_offset * going_left,
            (i_p_y[12] + i_p_y[8]) / 2 - annot_offset * going_down
        )

        arrow_3_annotate_coord = (
            (i_p_x[8] + i_p_x[5]) / 2 + annot_offset * going_left,
            (i_p_y[8] + i_p_y[5]) / 2 - annot_offset * going_down
        )

        ax.annotate("", xy=(i_p_x[12], i_p_y[12]), xytext=(i_p_x[18], i_p_y[18]), arrowprops=dict(arrowstyle="->", linewidth=2))
        ax.annotate("", xy=(i_p_x[8], i_p_y[8]), xytext=(i_p_x[12], i_p_y[12]), arrowprops=dict(arrowstyle="->", linewidth=2))
        ax.annotate("", xy=(i_p_x[5], i_p_y[5]), xytext=(i_p_x[8], i_p_y[8]), arrowprops=dict(arrowstyle="->", linewidth=2))

        ax.annotate("$\Delta \\vec{x}_i$", xy=arrow_1_annotate_coord, fontsize=fontsize)
        ax.annotate("$\Delta \\vec{x}_{i+1}$", xy=arrow_2_annotate_coord, fontsize=fontsize)
        ax.annotate("$\Delta \\vec{x}_{i+2}$", xy=arrow_3_annotate_coord, fontsize=fontsize)

        ax.grid(b=True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        max_x = np.hstack((adversary_points[:, 0], victim_points[:, 0])).max()
        min_x = np.hstack((adversary_points[:, 0], victim_points[:, 0])).min()
        max_y = np.hstack((adversary_points[:, 1], victim_points[:, 1])).max()
        min_y = np.hstack((adversary_points[:, 1], victim_points[:, 1])).min()
        print(max_x, min_x, max_y, min_y)
        ax.set_xlim(min_x - 2, max_x+2)
        ax.set_ylim(min_y - 1, max_y+1)

        # Shrink current axis's height by 5% on the bottom to fit legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.95])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.0), scatterpoints=1,
                  fancybox=True, shadow=True, ncol=2, fontsize=fontsize, handlelength=None, handletextpad=0)
        plt.tight_layout()

        plt.savefig("./plots/cm_explained.pdf", bbox_inches="tight")

