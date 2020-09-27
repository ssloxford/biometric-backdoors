import numpy as np
import os
from face import models
from scipy import misc
import facenet
from scipy import stats
from sklearn.svm import OneClassSVM
from keras.preprocessing import image
from keras import backend as K
import cv2

def load_data(folder_base, user, model_name=None, **kwargs):
    img_paths = get_image_paths(folder_base, user)
    if model_name is None:
        images = []
        for img_p in img_paths:
            images.append(misc.imread(img_p))
        return np.array(images)
    elif model_name in ["facenet-casia", "facenet-vgg2"]:
        assert "160" in folder_base
        images = facenet.load_data(img_paths, False, False, 160, do_prewhiten=True)
        return np.array(images)
    elif model_name in ["vgg16-lfw"]:
        assert "224" in folder_base
        images = []
        for i in img_paths:
            img = image.load_img(i, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x, version=1)
            x = np.squeeze(x)
            images.append(x)
        return np.array(images)
    elif model_name in ["vggresnet-vgg2"]:
        assert "224" in folder_base
        images = []
        for i in img_paths:
            img = image.load_img(i, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x, version=2)
            x = np.squeeze(x)
            images.append(x)
        return np.array(images)
    else:
        raise NotImplementedError("model %s is not implemented")


def undo_vggresnet(xs, **kwargs):
   assert xs.shape[-1] == 3
   x_temp = np.copy(xs)
   x_temp[..., 2] += 131.0912
   x_temp[..., 1] += 103.8827
   x_temp[..., 0] += 91.4953
   x_temp = x_temp[..., ::-1]
   return x_temp


def undo_vgg16(xs, **kwargs):
    # assert channels are last
    assert xs.shape[-1] == 3
    x_temp = np.copy(xs)
    x_temp[..., 2] += 129.1863
    x_temp[..., 1] += 104.7624
    x_temp[..., 0] += 93.5940
    x_temp = x_temp[..., ::-1]
    return x_temp


def undo_preprocessing(xs, name, **kwargs):
    assert name in ["vggresnet-vgg2", "vgg16-lfw", "facenet-casia", "facenet-vgg2"]
    assert xs.shape[-1] == 3
    functions = {
        "vgg16-lfw": undo_vgg16,
        "vggresnet-vgg2": undo_vggresnet,
        "facenet-casia": undo_facenet,
        "facenet-vgg2": undo_facenet 
    }
    return functions[name](xs, **kwargs)


def undo_facenet(xs, **kwargs):
    assert xs.shape[-1] == 3
    assert xs.min() <= 0 and xs.max() >=0
    x_temp = []
    for x in xs:
        x -= x.min()
        assert x.min() >= 0
        x = x/x.max()
        x = x * 255.0
        x_temp.append(x)
    return np.array(x_temp, dtype=int)



def do_preprocessing(xs, name, **kwargs):
    assert name in ["vggresnet-vgg2", "vgg16-lfw", "facenet-casia", "facenet-vgg2"]
    assert xs.shape[-1] == 3

    if name == "vggresnet-vgg2":
        return preprocess_input(xs, version=2)
    elif name == "vgg16-lfw":
        return preprocess_input(xs, version=1)
    elif name == "facenet-vgg2":
        return facenet.prewhiten(xs)
    elif name == "facenet-casia":
        return facenet.prewhiten(xs)
    else:
        raise Exception("Not impl")


def preprocess_input(x, data_format=None, version=1):

    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp


def get_meta_folder(config, model_name):
    assert model_name in config["models"]
    assert "face_padding" in config["models"][model_name]
    assert "face_padding" in config["models"][model_name]
    image_size = config["models"][model_name]["image_size"]
    face_padding = config["models"][model_name]["face_padding"]
    return "meta-%d-%d" % (image_size, face_padding)


def filter_inliers(emb, outliers_fraction, max_samples=1000):
    clf = OneClassSVM(nu=0.5, kernel="linear")
    clf.fit(emb)
    scores_pred = clf.decision_function(emb).flatten()
    n_samples = emb.shape[0]
    n_to_keep = min(max_samples, n_samples - int(outliers_fraction*n_samples))
    #print("ME", n_samples, n_to_keep, outliers_fraction)
    indexes = np.arange(0, n_samples, dtype=int)
    scores_i = np.argsort(scores_pred)[::-1]
    threshold = scores_pred[scores_i][n_to_keep]
    #print("THRESH", threshold)
    # threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
    i_mask = scores_pred > threshold
    return i_mask


def get_image_paths(aligned_f, user_id):
    image_paths = os.listdir(os.path.join(aligned_f, user_id))
    # get full path
    image_paths = map(lambda x: os.path.join(aligned_f, user_id, x), image_paths)
    # remove directories
    image_paths = filter(lambda x: not os.path.isdir(x), image_paths)
    # remove non-pngs
    image_paths = sorted(filter(lambda x: x.endswith(".png") or x.endswith(".jpg"), image_paths))
    return image_paths


def get_inliers_mask(meta_f, user, model):
    inliers_file = os.path.join(meta_f, user, "inliers_%s.csv" % model)
    assert os.path.isfile(inliers_file)
    # this contains absolute paths
    images_paths = get_image_paths(meta_f, user)
    # this contains relative paths
    images_paths = list(map(lambda x: x.split(os.path.sep)[-1], images_paths))
    inliers = np.loadtxt(inliers_file, delimiter=",", dtype=str)
    mask = np.array([True if x in inliers else False for x in images_paths], dtype=bool)
    return mask


def resize_images(images_in, height, width, interp="bilinear"):
    """
    :param images_in: numpy array containing input images to resize. Must be 4D array
    :param width: final width of the image
    :param height: final height of the image
    :return:
    """
    assert images_in.ndim == 4
    images_out = []
    if images_in.shape[1] == height and images_in.shape[2] == width:
        images_out = images_in
    else:
        for i in range(images_in.shape[0]):
            resized = misc.imresize(images_in[i], (width, height), interp=interp)
            # resized = np.array(Image.fromarray(images_in[i]).resize((width, height), Image.BILINEAR))
            images_out.append(resized)
        images_out = np.array(images_out)
    return images_out


model_names = {
    "facenet-vgg2": models.FacenetVGGFace2ResnetV1,
    "facenet-casia": models.FacenetCasiaResnetV1,
    "vggresnet-vgg2": models.VGGResNet,
    "vgg16-lfw": models.VGG16,
}
