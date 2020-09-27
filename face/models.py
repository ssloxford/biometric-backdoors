import os
import re

import numpy as np
import tensorflow as tf

import facenet
from cleverhans.model import Model
from deepface.recognizers.recognizer_resnet import FaceRecognizerResnet
from deepface.recognizers.recognizer_vgg import FaceRecognizerVGG


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


class VGG16(Model):
    input_shape = (224, 224, 3)
    output_shape = 4096
    class_name = "vgg16-lfw"
    input_mode = "BGR"

    def __init__(self, **kwargs):
        super(VGG16).__init__()
        print("[INFO] - Loading %s model" % self.class_name)
        self.recognizer = FaceRecognizerVGG(custom_db=None)
        self.graph = tf.get_default_graph()
        self.face_input = self.recognizer.input_node
        self.logits = self.recognizer.network["fc7"]
        self.embedding_output = tf.math.l2_normalize(tf.reshape(self.logits, (-1, self.output_shape)), axis=1, name="embedding_output")
        print("__init__, emb_out", self.embedding_output)
        self.class_output = self.recognizer.network["prob"]
        self.pt = tf.placeholder(dtype=tf.bool, shape=())
        feed_dict = {
            self.face_input: np.zeros(shape=(2,) + VGG16.input_shape),
            self.pt: False
        }
        self.persistent_sess = self.recognizer.persistent_sess

    def convert_to_classifier(self):
        self.target_embedding_input = tf.placeholder(tf.float32, shape=(None, VGG16.output_shape))
        print("convert_to_classifier, emb_out", self.embedding_output)
        distance = tf.reduce_sum(tf.square(self.embedding_output - self.target_embedding_input), axis=1)
        self.distance = distance
        threshold = 0.01
        score = tf.where(distance > threshold, 0.5 + ((distance - threshold) * 0.5) / (4.0 - threshold),
                         0.5 * distance / threshold)
        reverse_score = 1.0 - score
        self.softmax_output = tf.transpose(tf.stack([reverse_score, score]))
        self.layer_names = []
        self.layers = []
        print("ctc, softmax_out", self.softmax_output)
        self.layers.append(self.softmax_output)
        self.layer_names.append('probs')

    def fprop(self, x, set_ref=False):
        return dict(zip(self.layer_names, self.layers))

    def predict(self, x, batch_size=30):
        n_batches = x.shape[0] // batch_size + min(x.shape[0] % batch_size, 1)
        emb = np.zeros(shape=(0, self.output_shape))
        for i in range(n_batches):
            embeddings = self.recognizer.persistent_sess.run(
                self.embedding_output,
                feed_dict={self.face_input: x[i * batch_size:(i + 1) * batch_size], self.pt: False}
            )
            emb = np.vstack((emb, embeddings))
        return emb


class VGGResNet(Model):
    input_shape = (224, 224, 3)
    output_shape = 2048
    class_name = "vggresnet-vgg2"
    input_mode = "BGR"
 
    def __init__(self, **kwargs):
        print("[INFO] - Loading %s model" % self.class_name)
        super(VGGResNet).__init__()
        self.recognizer = FaceRecognizerResnet(custom_db=None)
        self.graph = tf.get_default_graph()
        self.face_input = self.recognizer.input_node
        print("__init__, input_node", self.face_input)
        self.logits = self.recognizer.network["feat"]
        print("__init__, logits", self.logits)
        self.embedding_output = tf.math.l2_normalize(tf.reshape(self.logits, (-1, self.output_shape)), axis=1, name="embedding_output")
        print("__init__, emb_out", self.embedding_output)
        self.class_output = self.recognizer.network["out"]
        print("__init__, class_output", self.class_output)
        self.pt = tf.placeholder(dtype=tf.bool, shape=())
        feed_dict = {
            self.face_input: np.zeros(shape=(2,) + VGGResNet.input_shape),
            self.pt: False
        }
        self.persistent_sess = self.recognizer.persistent_sess

    def convert_to_classifier(self):
        self.target_embedding_input = tf.placeholder(tf.float32, shape=(None, VGGResNet.output_shape))
        print("convert_to_classifier, emb_out", self.embedding_output)
        distance = tf.reduce_sum(tf.square(self.embedding_output - self.target_embedding_input), axis=1)
        self.distance = distance
        threshold = 0.01
        score = tf.where(distance > threshold, 0.5 + ((distance - threshold) * 0.5) / (4.0 - threshold),
                         0.5 * distance / threshold)
        reverse_score = 1.0 - score
        self.softmax_output = tf.transpose(tf.stack([reverse_score, score]))
        self.layer_names = []
        self.layers = []
        print("ctc, softmax_out", self.softmax_output)
        self.layers.append(self.softmax_output)
        self.layer_names.append('probs')

    def fprop(self, x, set_ref=False):
        return dict(zip(self.layer_names, self.layers))
 
    def predict(self, x, batch_size=30):
        n_batches = x.shape[0] // batch_size + min(x.shape[0] % batch_size, 1)
        emb = np.zeros(shape=(0, self.output_shape))
        for i in range(n_batches):
            embeddings = self.recognizer.persistent_sess.run(
                self.embedding_output,
                feed_dict={self.face_input: x[i * batch_size:(i + 1) * batch_size], self.pt: False}
            )
            # embeddings = embeddings.reshape(-1, self.output_shape)
            # embeddings = normalize(embeddings)
            emb = np.vstack((emb, embeddings))
        return emb



class FacenetVGGFace2ResnetV1(Model):
    input_shape = (160, 160, 3)
    output_shape = 512
    class_name = "facenet-vgg2"
    input_mode = "RGB"

    def __init__(self, models_folder, **kwargs):
        super(FacenetVGGFace2ResnetV1).__init__()
        print("[INFO] - Loading %s model" % self.class_name)
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.persistent_sess = tf.Session(config=config)

        facenet.load_model(os.path.join(models_folder, "facenet_vggface2_resnetv1", "20180402-114759.pb"))
        self.graph = tf.get_default_graph()
        self.persistent_sess = tf.Session(graph=self.graph, config=config)

        self.face_input = self.graph.get_tensor_by_name("input:0")
        self.embedding_output = self.graph.get_tensor_by_name("embeddings:0")
        self.pt = self.graph.get_tensor_by_name("phase_train:0")

        # warm up
        feed_dict = {
            self.face_input: np.zeros(shape=(2,) + FacenetVGGFace2ResnetV1.input_shape), self.pt: False
        }
        self.persistent_sess.run(tf.global_variables_initializer())
        _, _ = self.persistent_sess.run([self.face_input, self.embedding_output], feed_dict=feed_dict)

    def convert_to_classifier(self):
        # Create target_embedding placeholder
        self.target_embedding_input = tf.placeholder(tf.float32, shape=(None, 512))

        # Squared Euclidean Distance between embeddings
        distance = tf.reduce_sum(tf.square(self.embedding_output - self.target_embedding_input), axis=1)
        self.distance = distance
        # distance = tf.reduce_sum(self.embedding_output - self.target_embedding_input, axis=1)

        # Convert distance to a softmax vector
        # 0.99 out of 4 is the distance threshold for the Facenet CNN
        threshold = 0.1
        score = tf.where(distance > threshold, 0.5 + ((distance - threshold) * 0.5) / (4.0 - threshold), 0.5 * distance / threshold)
        reverse_score = 1.0 - score
        self.softmax_output = tf.transpose(tf.stack([reverse_score, score]))

        # Save softmax layer
        self.layer_names = []
        self.layers = []
        self.layers.append(self.softmax_output)

        self.layer_names.append('probs')

    def fprop(self, x, set_ref=False):
        return dict(zip(self.layer_names, self.layers))

    def predict(self, x):
        embeddings = self.persistent_sess.run(
            self.embedding_output,
            feed_dict={self.face_input: x, self.pt: False}
        )
        return embeddings


class FacenetCasiaResnetV1(Model):
    input_shape = (160, 160, 3)
    output_shape = 512
    class_name = "facenet-casia"
    input_mode = "RGB"

    def __init__(self, models_folder, **kwargs):
        super(FacenetCasiaResnetV1).__init__()
        print("[INFO] - Loading %s model" % self.class_name)
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.persistent_sess = tf.Session(config=config)

        facenet.load_model(os.path.join(models_folder, "facenet_casia_resnetv1", "20180408-102900.pb"))
        self.graph = tf.get_default_graph()
        self.persistent_sess = tf.Session(graph=self.graph, config=config)

        self.face_input = self.graph.get_tensor_by_name("input:0")
        self.embedding_output = self.graph.get_tensor_by_name("embeddings:0")
        self.pt = self.graph.get_tensor_by_name("phase_train:0")

        # warm up
        feed_dict = {
            self.face_input: np.zeros(shape=(2,) + FacenetCasiaResnetV1.input_shape), self.pt: False
        }
        self.persistent_sess.run(tf.global_variables_initializer())
        _, _ = self.persistent_sess.run([self.face_input, self.embedding_output], feed_dict=feed_dict)

    def convert_to_classifier(self):
        # Create target_embedding placeholder
        self.target_embedding_input = tf.placeholder(tf.float32, shape=(None, 512))

        # Squared Euclidean Distance between embeddings
        distance = tf.reduce_sum(tf.square(self.embedding_output - self.target_embedding_input), axis=1)
        self.distance = distance

        # Convert distance to a softmax vector
        # 0.99 out of 4 is the distance threshold for the Facenet CNN
        threshold = 0.1
        score = tf.where(distance > threshold, 0.5 + ((distance - threshold) * 0.5) / (4.0 - threshold),
                         0.5 * distance / threshold)
        reverse_score = 1.0 - score
        self.softmax_output = tf.transpose(tf.stack([reverse_score, score]))

        # Save softmax layer
        self.layer_names = []
        self.layers = []
        self.layers.append(self.softmax_output)

        self.layer_names.append('probs')

    def fprop(self, x, set_ref=False):
        return dict(zip(self.layer_names, self.layers))

    def predict(self, x):
        embeddings = self.persistent_sess.run(
            self.embedding_output,
            feed_dict={self.face_input: x, self.pt: False}
        )
        return embeddings


