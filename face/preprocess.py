import argparse
import json
import os
import shutil
import sys
import time
import cv2
import dlib
import imutils
import numpy as np
import pandas as pd
from imutils import face_utils
from scipy import stats
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

from face import face_utils as fut
from face import models
import tensorflow as tf


def align(input_dir, output_dir, image_size, margin, facenet_path):
    if os.path.isdir(output_dir):
        print("Aligned dataset %s exists already,\ndo you want to recreate it?[y/n]" % output_dir)
        a = input("")
        if a == "y":
            shutil.rmtree(output_dir)
        else:
            return False

    os.chdir(facenet_path)
    os.system("python {} {} {} --image_size {} --margin {}".format(
        os.path.join(facenet_path, "src", "align", "align_dataset_mtcnn.py"),
        input_dir,
        output_dir,
        image_size,
        margin
    ))


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def compute_outliers(meta_f, models_folder, model_name, outliers_fraction, landmarked_only):
    """

    :param meta_f:
    :param model:
    :param models_folder:
    :return:
    """
    assert model_name in fut.model_names.keys()
    id_list = os.listdir(meta_f)
    id_list = sorted(filter(lambda x: os.path.isdir(os.path.join(meta_f, x)), id_list))
    clf = OneClassSVM(nu=0.5, kernel="rbf", gamma=0.1)
    # clf = EllipticEnvelope(contamination=outliers_fraction)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = fut.model_names[model_name](models_folder)
            for user in id_list:
                print(user)
                # compute embeddings
                images = fut.load_data(meta_f, user, model_name)
                X = model.predict(images)
                image_paths = fut.get_image_paths(meta_f, user)
                # take only relative path
                inliers_images = np.array(list(map(lambda x: x.split(os.path.sep)[-1], image_paths)))
                assert X.shape[0] == len(image_paths)
                if landmarked_only:
                    landmarked = pd.read_csv(os.path.join(meta_f, user, "landmarks.csv"), index_col=0)
                    assert X.shape[0] == len(image_paths) == landmarked.shape[0]
                    nan_mask = np.isnan(landmarked.values).any(axis=1)
                    X = X[~nan_mask]
                    inliers_images = inliers_images[~nan_mask]

                # fit classifier to data
                clf.fit(X)
                # compute distance to decision function
                scores_pred = clf.decision_function(X).flatten()
                # check at what treshold we find <outliers_fraction>% of outliers
                threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
                inliers_mask = (scores_pred >= threshold)
                inliers_images = inliers_images[inliers_mask]
                np.savetxt(os.path.join(meta_f, user, "inliers_%s.csv" % model_name), inliers_images, delimiter=",", fmt="%s")
    return True


def compute_embeddings(meta_f, model_name, models_folder):
    """
    :param meta_f: output folder which will contain embeddings
    :param model: model to use, either of ["facenet", "openface"]
    :return:
    """
    assert model_name in fut.model_names.keys()

    id_list = os.listdir(meta_f)
    id_list = sorted(filter(lambda x: os.path.isdir(os.path.join(meta_f, x)), id_list))

    # this loads the model calling the loader function defined above in model_names
    model = fut.model_names[model_name](models_folder)

    for id in id_list:
        print("[INFO] - user %s" % id)
        image_paths = fut.get_image_paths(meta_f, id)

        images = fut.my_data_load(image_paths, model.input_shape[0])
        embeddings = model.predict(images)

        os.makedirs(os.path.join(meta_f, id), exist_ok=True)
        np.savetxt(os.path.join(meta_f, id, "emb_%s.csv" % model.class_name), embeddings, delimiter=",")


def compute_landmarks(aligned_f, predictor_path, output_folder, vis):
    """
    # p0 = left part of left eye
    # p1 = right part of left eye
    # p2 = right part of right eye
    # p3 = left part of right eye
    # p4 = nose
    :param aligned_f:
    :param predictor_path:
    :param output_folder:
    :param vis:
    :return:
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    id_list = os.listdir(aligned_f)
    id_list = sorted(filter(lambda x: os.path.isdir(os.path.join(aligned_f, x)), id_list))
    # id_list = np.array(id_list)[np.array(id_list) > "n006908"].tolist()
    for user in id_list:
        os.makedirs(os.path.join(output_folder, user), exist_ok=True)
        image_paths = fut.get_image_paths(aligned_f, user)
        landmarks = []
        image_filenames = []
        for j, image_path in enumerate(image_paths):
            frame = cv2.imread(image_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            rects = detector(gray, 0)
            if len(rects) == 0:
                print("[WARNING] - %s, %s, face not found" % (user, image_paths[j]))
            else:
                image_name = image_path.split(os.path.sep)[-1]
                image_filenames.append(image_name)
                rect = rects[0]
                lms = predictor(gray, rect)
                lms = face_utils.shape_to_np(lms)
                ll_eye = lms[0]
                rr_eye = lms[2]
                # then we do a rotation to make the eyes horizontal
                x = ll_eye[0] - rr_eye[0]
                y = ll_eye[1] - rr_eye[1]
                assert x > 0
                angle_degrees = np.degrees(np.arctan(y/x))
                M_inv = cv2.getRotationMatrix2D((frame.shape[0]/2, frame.shape[1]/2), angle_degrees, 1)
                lms = np.hstack([lms, np.ones(shape=(len(lms), 1))])
                lms = M_inv.dot(lms.T).T.astype(int)
                rotated_frame = imutils.rotate(frame, angle_degrees)
                cv2.imwrite(os.path.join(output_folder, user, image_name), rotated_frame)
                landmarks.append(lms)

        landmarks = np.array(landmarks).astype(float)
        landmarks = landmarks.reshape((landmarks.shape[0], -1))
        columns_x = ["x%d" % v for v in range(5)]
        columns_y = ["y%d" % v for v in range(5)]
        landmarks = pd.DataFrame(landmarks, index=image_filenames, columns=[val for pair in zip(columns_x, columns_y) for val in pair])
        os.makedirs(os.path.join(output_folder, user), exist_ok=True)
        landmarks.to_csv(os.path.join(output_folder, user, "landmarks.csv"))


description = """
Align a dataset of facial images and computes landmarks.
 * Alignment requires align_dataset_mtcnn.py script.
 * landmark detection uses dlib implementation of face detection and landmark detection (shape_predictor_5_face_landmarks.dat)

Example usage  
 python preprocess.py -a ~/f/dataset/vggface2_test 160 36
 python preprocess.py -l shape_predictor_5_face_landmarks.dat ~/f/dataset/aligned-160-36 ~/f/dataset/meta-160-36 0

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-a", "--align", action="store", nargs=3, metavar=('source_dir', "image_size", "margin"), help="Whether to align dataset")
    parser.add_argument("-l", "--landmarks", action="store", nargs=4, metavar=("predictor", "aligned_dir", "meta_dir", "visualize"), help="Whether to compute landmarks")
    parser.add_argument('-c', '--config', action='store', type=str, default="./config.json")

    args = parser.parse_args()
    conf = json.load(open(args.config, "r"))
    model_folder = os.path.join(conf["paths"]["data"], "models")

    if args.align is not None:
        # then align the dataset
        input_f, image_size, image_margin = args.align
        base_f = os.path.sep.join(os.path.abspath(input_f).split(os.path.sep)[:-1])
        output_f = os.path.join(base_f, "aligned-%s-%s" % (image_size, image_margin))
        a = input("Saving in folder %s\nIs that all right?[y/n]" % output_f)
        if a == "y":
            align(input_f, output_f, int(image_size), int(image_margin), conf["paths"]["facenet"])

    if args.landmarks is not None:
        predictor_name, aligned_f, meta_f, visualize = args.landmarks
        predictor_path = os.path.join(model_folder, predictor_name)
        assert visualize in ["0", "1"]
        image_size_s = aligned_f.split(os.path.sep)[-1].split("-")[1]
        image_size_t = meta_f.split(os.path.sep)[-1].split("-")[1]
        face_pad_s = aligned_f.split(os.path.sep)[-1].split("-")[2]
        face_pad_t = meta_f.split(os.path.sep)[-1].split("-")[2]
        assert image_size_s == image_size_t
        assert face_pad_s == face_pad_t
        compute_landmarks(aligned_f, predictor_path, meta_f, bool(int(visualize)))

