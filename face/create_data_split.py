import argparse
import json
import os
import sys

import numpy as np
import sklearn
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle

from matchers.matchers import get_matchers_list
import face_utils as fut


def load_all_dev_users(dev_users, model, meta_dir):
    print("pre-loading all embeddings")
    loaded = {}
    for j, u in enumerate(dev_users):
        X_usr = fut.load_data(meta_dir, u, model_name=model.class_name)
        x_usr = model.predict(X_usr)
        inliers_mask = fut.filter_inliers(x_usr, args.outliers_fraction)
        x_usr = x_usr[inliers_mask]
        loaded[u] = x_usr
        sys.stdout.write("\r%d/%d" % (j+1, len(dev_users)))
        sys.stdout.flush()
    sys.stdout.write("\n")
    return loaded


if __name__ == "__main__":

    random_seed = 42

    np.random.seed(random_seed)

    parser = argparse.ArgumentParser(
        description="Split training and test data\n"
                    "python create_data_split.py -m facenet-vgg -n 10"
    )
    parser.add_argument("-m", "--model_name", action="store", type=str, choices=["openface", "facenet-vgg2", "facenet-casia", "vgg16-lfw", "vggresnet-vgg2"])
    parser.add_argument('-n', '--n_training_samples', action='store', required=True, type=int)
    parser.add_argument('-d', '--n_dev_users', action='store', default=100, type=int)
    parser.add_argument('--n_negative_users', action='store', default=100, type=int)
    parser.add_argument('-o', '--outliers_fraction', action='store', default=0.1, type=float)
    parser.add_argument('-c', '--config', action='store', type=str, default="./config.json")

    args = parser.parse_args()

    assert args.n_negative_users <= args.n_dev_users

    conf_face = json.load(open(args.config, "r"))

    data_path = conf_face["paths"]["data"]
    meta_dir = fut.get_meta_folder(config=conf_face, model_name=args.model_name)
    print(meta_dir)
    meta_dir = os.path.join(data_path, "dataset", meta_dir)
    users = sorted(list(filter(lambda x: os.path.isdir(os.path.join(meta_dir, x)), os.listdir(meta_dir))))
    users = shuffle(users, random_state=random_seed)
    n_dev_users = args.n_dev_users
    n_test_users = len(users) - n_dev_users
    development_users, test_users = users[:n_dev_users], users[n_dev_users:]
    adversaries, victims = sorted(test_users[:n_test_users//2]), sorted(test_users[n_test_users//2:])

    all_preds = {}
    all_labels = []
    results_dir = os.path.join(data_path, "results")
    outf = os.path.join(results_dir, "{}_{}".format(args.model_name, args.n_training_samples))
    os.makedirs(outf, exist_ok=True)

    ccc = dict()
    ccc["dev_users"] = dict()
    ccc["test_users"] = dict()
    ccc["adversaries"] = adversaries
    ccc["victims"] = victims

    clfs = get_matchers_list(mode="flat") + get_matchers_list(mode="sigmoid")
    for clf in clfs:
        all_preds[clf.get_name()] = []

    models_folder = os.path.join(conf_face["paths"]["data"], "models")

    model = fut.model_names[args.model_name](models_folder=models_folder)
    loaded = load_all_dev_users(development_users, model, meta_dir)
    output_size = model.output_shape

    for i, u in enumerate(development_users):
        print("dev user %s, (%d/%d)" % (u, i+1, len(development_users)))
        ccc["dev_users"][u] = dict()
        x_usr = loaded[u]
        n_samples = x_usr.shape[0]
        chosen_train = np.random.choice(range(n_samples), args.n_training_samples, replace=False)
        mask_array = np.zeros(x_usr.shape[0], dtype=bool)
        mask_array[chosen_train] = True
        x_train = x_usr[mask_array]
        x_test_usr = x_usr[~mask_array]
        x_test_others = np.zeros(shape=(0, output_size), dtype=float)

        ccc["dev_users"][u]["chosen_train"] = list(map(int, chosen_train))
        ccc["dev_users"][u]["n_samples"] = int(n_samples)

        _other_users = np.random.choice(n_dev_users, args.n_negative_users, replace=False)
        _other_users = np.array(users, dtype=str)[_other_users]
        for j, v in enumerate(_other_users):
            if v == u:
                continue
            _x_vusr = loaded[v]
            _chosen = np.random.choice(range(_x_vusr.shape[0]), 10, replace=False)
            x_test_others = np.vstack((x_test_others, _x_vusr[_chosen]))

        x_test = np.vstack((x_test_usr, x_test_others))
        y_test = np.hstack((np.ones(x_test_usr.shape[0]), np.zeros(x_test_others.shape[0])))

        for clf in clfs:
            clf.fit(x_train)
            all_preds[clf.get_name()].extend(clf.predict(x_test))

        all_labels.extend(y_test)

    for clf in clfs:
        all_preds[clf.get_name()] = np.array(all_preds[clf.get_name()]).flatten()

    all_labels = np.array(all_labels, dtype=int)

    fpr, tpr, thr, roc_auc, eer_thr, eer = {}, {}, {}, {}, {}, {}
    for clf in clfs:
        cn = clf.get_name()
        fpr[cn], tpr[cn], thr[cn] = roc_curve(all_labels, all_preds[clf.get_name()], pos_label=1)
        roc_auc[cn] = auc(fpr[cn], tpr[cn])
        fnr = 1 - fpr[cn]
        eer[cn] = brentq(lambda x: 1. - x - interp1d(fpr[cn], tpr[cn])(x), 0., 1.)
        eer_thr[cn] = interp1d(fpr[cn], thr[cn])(eer[cn])
        print(cn, eer[cn], roc_auc[cn])
        ccc["eer_%s" % cn] = eer[cn]
        ccc["thr_%s" % cn] = float(eer_thr[cn])
        ccc["auc_%s" % cn] = float(roc_auc[cn])

    json.dump(ccc, open(os.path.join(outf, "ccc.json"), "w"), indent=True)

    # first compute the train mask for the test users
    # second find and save a list of possible targets
    # that are accepted by the classifier
    for i, u in enumerate(test_users):
        print("test user %s, (%d/%d)" % (u, i+1, len(test_users)))
        ccc["test_users"][u] = dict()

        X_usr = fut.load_data(meta_dir, u, model_name=model.class_name)
        x_usr = model.predict(X_usr)
        inliers_mask = fut.filter_inliers(x_usr, args.outliers_fraction)
        x_usr = x_usr[inliers_mask]

        n_samples = x_usr.shape[0]
        chosen_train = np.random.choice(range(n_samples), args.n_training_samples, replace=False)
        mask_array = np.zeros(x_usr.shape[0], dtype=bool)
        mask_array[chosen_train] = True
        x_train = x_usr[mask_array]
        x_test_usr = x_usr[~mask_array]
        x_test_others = np.zeros(shape=(0, output_size), dtype=float)
        ccc["test_users"][u]["chosen_train"] = list(map(int, chosen_train))
        ccc["test_users"][u]["n_samples"] = int(n_samples)

        for clf in clfs:
            clf.fit(x_train)

        chosen_targets = []
        j = 0
        # reorder samples
        shuffled = sklearn.utils.shuffle(range(x_usr.shape[0]))
        while j < x_usr.shape[0]:
            # check whether j-th element is within threshold for all classifiers
            distances = {}
            for clf in clfs:
                d_clf = clf.predict(x_usr[shuffled[j]][np.newaxis, :])[0]
                # print(clf.get_name(), d_clf, ccc["thr_%s" % clf.get_name()])
                distances[clf.get_name()] = d_clf >= ccc["thr_%s" % clf.get_name()]
            # print(distances)
            
            if np.array(list(distances.values()), dtype=bool).all() and shuffled[j] not in chosen_train:
                # print(j)
                chosen_targets.append(shuffled[j])
            j += 1
        ccc["test_users"][u]["chosen_targets"] = chosen_targets

    json.dump(ccc, open(os.path.join(outf, "ccc.json"), "w"), indent=True)



