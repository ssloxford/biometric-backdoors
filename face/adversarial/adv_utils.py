import itertools
import os
import sys
import warnings
import cv2
import keras
import numpy as np
import scipy.misc
import tensorflow as tf
from keras import backend
from scipy.linalg import norm
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import euclidean
import facenet
from cleverhans.model import Model
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.path.join(".."))
from face import face_utils as fut


def load_args(model_name, config_dict, script, args):

    args_dict = vars(args)
    defaults_dict = config_dict["models"][model_name][script]
    for key in args_dict.keys():
        if args_dict[key] is None:
            args_dict[key] = defaults_dict[key]
    args = argparse.Namespace(**args_dict)
    return args


def _smooth_one_and_go(X_adv_g, gmasks, sigma=0.5):
    assert gmasks.ndim == 2
    shape = X_adv_g.shape[1:]
    # reshape into (N,n_features)
    X_adv_g = X_adv_g.reshape(X_adv_g.shape[0], -1)
    # remove things outside of glasses
    # print(X_adv_g.shape, gmasks.shape)
    glasses = X_adv_g * gmasks
    # back to original shape
    glasses = glasses[0].reshape(shape)
    # make outside of glasses of similar color to avoid ruining the smoothing
    glasses[glasses == 0] = glasses[glasses != 0].mean()
    glasses = gaussian_filter(glasses, sigma)
    # extract color
    color_only = glasses.flatten()[gmasks[0]]
    X_adv_g = X_adv_g.reshape((-1,) + shape)
    X_adv_g = _local_apply_stuff(X_adv_g, color_only, gmasks, shape)
    return X_adv_g

def _local_apply_stuff(images, glasses_color, glasses_masks, base_shape):
    applied = []
    for i in range(images.shape[0]):
        image = images[i]
        gmask = glasses_masks[i].flatten()
        glasses_this = np.zeros(shape=base_shape).flatten()
        glasses_this[gmask] = glasses_color
        glasses_this = glasses_this.reshape(images.shape[1:])
        adv = apply_glasses(image[np.newaxis, :], glasses_this[np.newaxis, :], gmask[np.newaxis, :])
        applied.append(adv[0])
    return np.array(applied)


def check_exists(fold, files, heur=""):
    for f in files:
        if not os.path.isfile(os.path.join(fold, heur, f)):
            return False
    return True


def get_other_users_embedding(users, this_user, user_folder, n=10):
    other_tset = np.zeros(shape=(0, 128), dtype=float)
    for j, v in enumerate(users):
        if v != this_user:
            _e_vusr = np.load(os.path.join(user_folder, v, "e.npy"))
            _chosen = np.random.choice(range(_e_vusr.shape[0]), n, replace=False)
            other_tset = np.vstack((other_tset, _e_vusr[_chosen]))
    return other_tset


def get_info_from_log(logfile):
    with open(logfile, "r") as f:
        data = f.read().replace('\n', '')
    prop = dict()
    info, changes = data.split("###start###")
    prop["adversary_id"] = info[info.find("adversary_id"):].split(":")[1].split("#")[0].rstrip(" ").lstrip(" ")
    prop["user_id"] = info[info.find("user_id"):].split(":")[1].split("#")[0].rstrip(" ").lstrip(" ")
    prop["theta"] = float(info[info.find("theta"):].split(":")[1].split("#")[0].rstrip(" ").lstrip(" "))
    prop["maxiters"] = int(info[info.find("maxiters"):].split(":")[1].split("#")[0].rstrip(" ").lstrip(" "))
    prop["n_pixels"] = int(info[info.find("n_pixels"):].split(":")[1].split("#")[0].rstrip(" ").lstrip(" "))
    prop["eer_thr_cnt"] = float(info[info.find("eer_thr_cnt"):].split(":")[1].split("#")[0].rstrip(" ").lstrip(" "))
    prop["eer_thr_mxm"] = float(info[info.find("eer_thr_mxm"):].split(":")[1].split("#")[0].rstrip(" ").lstrip(" "))
    prop["eer_thr_svm"] = float(info[info.find("eer_thr_svm"):].split(":")[1].split("#")[0].rstrip(" ").lstrip(" "))
    return prop, changes.split("###end###")[0]


def process_log_file(logfile, adversary, user):
    if not os.path.isfile(logfile):
        raise RuntimeError("There is no generation log at %s" % logfile)
    if file_len(logfile) < 20:
        raise RuntimeError("Too few lines in %s" % logfile)
    info, changes = get_info_from_log(logfile)
    if adversary != info["adversary_id"]:
        raise RuntimeError("File contains adversary_id=%s but is located in the wrong folder %s" % (info["adversary_id"], logfile))
    if user != info["user_id"]:
        raise RuntimeError("File contains user_id=%s but is located in the wrong folder %s" % (info["user_id"], logfile))
    return info, changes


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def split_edits(changes):
    changes = changes.replace("[[", "[")
    changes = changes.replace("]]", "]")
    start = changes.find("D,[")
    while start >= 0:
        # find next ']'
        end = changes.find("]", start, len(changes))
        changes = changes[:start] + changes[end + 1:]
        start = changes.find("D,")
    changes = changes.replace("\n", "")
    changes = changes.replace("+", "#+")
    changes = changes.replace("-", "#-")
    changes = changes.split("#")[1:]
    changes = list(map(lambda x: x.split(","), changes))

    p_changes, n_changes = [], []
    for i, c in enumerate(changes):
        sign, indexes = c
        indexes = list(map(int, indexes[1:-1].split(" ")))
        if sign == "+":
            p_changes.extend(indexes)
        if sign == "-":
            n_changes.extend(indexes)

    return np.array([p_changes, n_changes], dtype=int).T


def gds_step(poisoned_samples, index_p, index_m, theta, clip_min, clip_max, direction, height, width, channels):
    mask_p = mask_from_index(index_p, height * width * channels)
    mask_m = mask_from_index(index_m, height * width * channels)
    poisoned_samples_flat = poisoned_samples.reshape(-1, height * width * channels)
    if direction == "f":
        poisoned_samples_flat[:, mask_p] += theta
        poisoned_samples_flat[:, mask_m] -= theta
    elif direction == "b":
        poisoned_samples_flat[:, mask_p] -= theta
        poisoned_samples_flat[:, mask_m] += theta
    else:
        raise RuntimeError("gds_step direction can only be 'f' or 'b'")
    poisoned_samples = poisoned_samples_flat.reshape(-1, height, width, channels)
    poisoned_samples = np.clip(poisoned_samples, clip_min, clip_max)
    return poisoned_samples


def gds_step_bobby(X_adv, X_adv_g, index_p, index_m, offsets, theta, clip_min, clip_max, glasses_masks):
    height, width, channels = X_adv_g.shape[1:]

    # first roll one pair of glasses to top left to apply changes
    glasses = (glasses_masks[0] * X_adv_g[0])
    glasses = np.roll(glasses, -offsets[0][0], axis=1)
    glasses = np.roll(glasses, -offsets[0][1], axis=0)
    glasses = glasses.reshape(-1)

    mask_p = mask_from_index(index_p, height * width * channels)
    mask_m = mask_from_index(index_m, height * width * channels)

    glasses[mask_p] += theta
    glasses[mask_m] -= theta

    X_adv_g = _local_apply_stuff(X_adv, glasses[glasses!=0], glasses_masks, X_adv.shape[1:])
    X_adv_g = np.clip(X_adv_g, clip_min, clip_max)
    return X_adv_g


def elem_wise_l2(x, y):
    assert x.ndim == 2
    ys = np.array(y)
    if y.ndim < 2:
        ys = np.tile(y, (x.shape[0], 1))
    d = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        d[i] = norm(ys[i] - x[i], 2)
    return d


def elem_wise_l1(x, y):
    assert x.ndim == 2
    ys = np.array(y)
    if y.ndim < 2:
        ys = np.tile(y, (x.shape[0], 1))
    d = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        d[i] = norm(ys[i] - x[i], 1)
    return d


class MyModelWrapper(Model):
    def __init__(self, model=None):
        super(MyModelWrapper, self).__init__()
        if model is None:
            raise ValueError('model argument must be supplied.')
        self.model = model
        self.keras_model = None

    def _get_softmax_name(self):
        for i, layer in enumerate(self.model.layers):
            cfg = layer.get_config()
            if layer.name == "softmax":
                return layer.name
        raise Exception("No softmax layers found")

    def _get_logits_name(self):
        softmax_name = self._get_softmax_name()
        softmax_layer = self.model.get_layer(softmax_name)
        if hasattr(softmax_layer, 'inbound_nodes'):
            warnings.warn(
                "Please update your version to keras >= 2.1.3; "
                "support for earlier keras versions will be dropped on "
                "2018-07-22")
            node = softmax_layer.inbound_nodes[0]
        else:
            node = softmax_layer._inbound_nodes[0]

        logits_name = node.inbound_layers[0].name
        return logits_name

    def get_logits(self, x, **kwargs):
        logits_name = self._get_logits_name()
        return self.get_layer(x, logits_name)

    def get_probs(self, x, **kwargs):
        name = self._get_softmax_name()
        return self.get_layer(x, name)

    def get_layer_names(self):
        layer_names = [x.name for x in self.model.layers]
        return layer_names

    def fprop(self, x, **kwargs):
        from keras.models import Model as KerasModel
        if self.keras_model is None:
            new_input = self.model.get_input_at(0)

            #out_layers = [x_layer.output for x_layer in self.model.layers]
            #self.keras_model = KerasModel(new_input, out_layers)

            out_layers = []
            for x_layer in self.model.layers:
                out_layers.append(x_layer.get_output_at(-1))
            out_layers = out_layers
            self.keras_model = KerasModel(new_input, out_layers)

        # and get the outputs for that model on the input x
        outputs = self.keras_model(x)

        # Keras only returns a list for outputs of length >= 1, if the model
        # is only one layer, wrap a list
        if len(self.model.layers) == 1:
            outputs = [outputs]

        # compute the dict to return
        fprop_dict = dict(zip(self.get_layer_names(), outputs))

        return fprop_dict


def euclidean2d(x, y):
    assert x.shape[0] == y.shape[0]
    out = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        out[i] = euclidean(x[i], y[i])
    return out.mean()


def all_perms():
    l = []
    for p in itertools.product([-2, -1, 0, 1, 2], repeat=2):
        l.append(p)
    return np.array(l)


def mask_from_index(indexes, n):
    _indexes = np.array(indexes)
    if _indexes.ndim > 1:
        mask = np.zeros(shape=(_indexes.shape[0], n), dtype=bool)
        for t in range(_indexes.shape[0]):
            mask[t][_indexes[t]] = True
    else:
        mask = np.zeros(n, dtype=bool)
        mask[_indexes] = True
    return mask


def adjust_saturation(img, value=-30):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:
        lim = 255 - value
        s[s > lim] = 255
        s[s <= lim] += value
    if value < 0:
        lim = 0 - value
        s[s <= lim] = 0
        s[s > lim] += np.uint8(value)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def adjust_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    if value < 0:
        lim = 0 - value
        v[v <= lim] = 0
        v[v > lim] += np.uint8(value)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def init_random_glasses(glasses_mask, v_min, v_max, index_of_blue_channel):
    assert glasses_mask.ndim == 3
    assert glasses_mask.dtype == np.bool
    assert v_min <= 0
    assert type(index_of_blue_channel) == int
    shape = glasses_mask.shape
    ra = np.random.uniform(0, 255, size=(shape[0], shape[1], 1))
    rg = np.random.uniform(0, 255, size=(shape[0], shape[1], 1))
    rb = np.random.uniform(0, 255, size=(shape[0], shape[1], 1))
    glasses = scipy.misc.imresize(np.concatenate((ra, rg, rb), axis=2), size=shape)
    glasses[:, :, index_of_blue_channel] = glasses[:,:, index_of_blue_channel]/4
    glasses = adjust_saturation(glasses.astype(np.uint8), value=-20)
    glasses = adjust_brightness(glasses.astype(np.uint8), value=+10)
    glasses = gaussian_filter(glasses, 1.0).astype(int)
    glasses = (glasses.flatten() * glasses_mask.flatten().astype(int)).reshape(shape)
    max_factor = glasses.max()/255
    glasses = (glasses - glasses.min())/(glasses.max()-glasses.min())*max_factor
    new_range = np.abs(v_max-v_min)
    glasses = glasses*new_range + v_min
    return glasses


def do_a_barrel_roll(old_pos, new_pos, g, mask, orig_shape):
    n_features = np.product(orig_shape)
    g = g.reshape((-1,) + orig_shape)
    mask = mask.reshape((-1,) + orig_shape)
    # now roll both directions
    shift_pos = new_pos - old_pos
    g = np.roll(g, shift=shift_pos[0], axis=1)
    mask = np.roll(mask, shift=shift_pos[0], axis=1)
    g = np.roll(g, shift=shift_pos[1], axis=2)
    mask = np.roll(mask, shift=shift_pos[1], axis=2)
    g = g.reshape((-1, n_features))
    mask = mask.reshape((-1, n_features))
    return new_pos, g, mask


def check_keras():
    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured to use the TensorFlow backend.")
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'th', temporarily setting to 'tf'")
    return True


def pad_3d_image(img, left, right, top, bottom, val=0.0):
    output_width = img.shape[1] + left + right
    output_height = img.shape[0] + top + bottom
    img = img.transpose((2, 1, 0))
    pads = ((left, right), (top, bottom))
    img_arr = np.ndarray((3, output_width, output_height), np.int)
    for i, x in enumerate(img):
        x_p = np.pad(x, pads, 'constant', constant_values=val)
        img_arr[i, :, :] = x_p
    return img_arr.transpose((2, 1, 0))


def get_glasses_with_dimensions(filepath, width, height):
    assert filepath.split(".")[-1] in ["jpg", "bmp"]
    glasses = scipy.misc.imread(filepath)
    glasses[glasses>=255/2.0] = 255
    glasses[glasses<255 / 2.0] = 0
    glasses = fut.resize_images(glasses[np.newaxis], width, height)[0]
    glasses[glasses >= 255 / 2.0] = 255
    glasses[glasses < 255 / 2.0] = 0
    g_mask = glasses == 0.0
    return np.abs(glasses-255), g_mask


def get_glasses(filepath):
    """
    :param filepath: filepath for the glasses file .jpg or anything
    :return: np array containing glasses, np array containing boolean mask with 1s where the glasses are
    """
    assert filepath.split(".")[-1] in ["jpg", "bmp"]
    glasses = scipy.misc.imread(filepath)
    # glasses = fut.resize_images(glasses, width, height)
    glasses[glasses>=255/2.0] = 255
    glasses[glasses<255 / 2.0] = 0
    g_mask = glasses == 0.0
    return glasses, g_mask


def get_all_glasses_masks(landmarks, glasses_path, image_shape):
    g_width = int(image_shape[1]*.72)
    g_height = int(image_shape[1]*.27)
    pad_left = int(image_shape[1] * .125)
    pad_top = int(image_shape[1] * .135)
    each_image_mask = []
    paddings = []
    for i in range(landmarks.shape[0]):
        lm = landmarks[i].astype(int)
        points = list(zip(lm[::2], lm[1::2]))
        glasses, gmask_1 = get_glasses_with_dimensions(glasses_path, g_width, g_height)
        left = max(0, points[2][0] - pad_left)
        right = max(0, image_shape[1] - left - glasses.shape[1])
        top = max(0, points[2][1] - pad_top)
        bottom = max(0, image_shape[0] - top - glasses.shape[0])
        while (left + right + g_width) > image_shape[1]:
            # print(left, right, g_width)
            if left > 0:
                left -= 1
                continue
            if right > 0:
                right -= 1
                continue

        gmask_1 = pad_3d_image(gmask_1, left, right, top, bottom, 0).astype(bool)
        each_image_mask.append(gmask_1)
        paddings.append([left, right, top, bottom])
    return np.array(each_image_mask), np.array(paddings)


def get_glasses_mask(glasses):
    assert glasses.ndim == 4
    assert glasses.dtype == np.int
    return glasses.reshape((glasses.shape[0], -1)) > 50.0


def apply_glasses(x_in, glasses_in, g_masks_in):
    """

    :param x_in:
    :param glasses_in:
    :param g_masks_in:
    :return:
    """
    # check that it's bool
    assert x_in.ndim == 4
    assert glasses_in.ndim == 4
    assert x_in.shape == glasses_in.shape
    assert g_masks_in.ndim == 2
    assert g_masks_in.dtype == np.bool
    #for x, g in zip(x_in, glasses_in):
        #print(x.min(), g.min(), x.max(), g.max())
        #assert x.min() <= g.min() and x.max() >= g.max()

    # flatten
    x_out = np.array(x_in).reshape((x_in.shape[0], -1))
    glasses_out = np.array(glasses_in).reshape((glasses_in.shape[0], -1))
    for i in range(x_out.shape[0]):
        x_out[i, g_masks_in[i]] = glasses_out[i, g_masks_in[i]]
    return x_out.reshape(x_in.shape)


def apply_glasses_single(x, glasses, gmask):
    """
    :param x: contains the samples where we'll overlay the glasses, shape (96, 96, 3)
    :param glasses: contains the glasses, shape (96*96*3)
    :param gmask: boolean mask for the glasses
    :return:
    """
    if gmask.ndim > 1:
        raise RuntimeError("Gmask should be flat")
    if gmask.dtype != np.bool:
        raise RuntimeError("gmask should be bool")
    orig_shape = x.shape
    adv_x = x.flatten()
    adv_x[gmask] = 0.0
    adv_x = adv_x.reshape(orig_shape)
    adv_x = adv_x + glasses
    scipy.misc.imshow(adv_x)
    return adv_x
