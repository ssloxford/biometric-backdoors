import matplotlib
import matplotlib.pyplot as plt
import os
import scipy
import scipy.stats
import numpy as np

plt.style.use('classic')
matplotlib.rc('font', family='CMU')
matplotlib.rc('font', serif='Serif')
matplotlib.rc('text', usetex='true')
matplotlib.rcParams.update({'font.size': 22})


model_to_names = {
    "facenet-vgg2": "FaceNet",
    "vggresnet-vgg2": "ResNet-50",
    "vgg16-lfw": "VGG16",
    "facenet-casia": "FaceNet-\\footnotesize{CASIA}"
}



colors = [
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "brown",
    "pink",
    "grey",
    "black",
    "yellow",
    "aqua"
]

markers = [
    "o",
    "s",
    "^",
    "v"
    "x",
    ">"
]

success_thresh = 0.3

clfs = ["cnt", "mxm"] + ["svm_%.2f" % nu for nu in np.arange(0.1, 1.0, 0.1)]


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.nanmean(a), scipy.stats.sem(a[~np.isnan(a)])
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def get_poisoining_rates(folder, adversaries, users, fname, heur="", ):
    dct = {clf: [] for clf in clfs}
    for i, clf in enumerate(clfs):
        for a in adversaries:
            for u in users:
                if os.path.isfile(os.path.join(folder, a, u, clf, heur, fname)):
                    l = np.loadtxt(os.path.join(folder, a, u, clf, heur, fname), delimiter=",")
                    l = l.sum(axis=1)/l.shape[1]
                    # print(l, os.path.join(f, a, u, clf, heur, "accepted_indexes.csv"))
                    dct[clf].append(l.tolist()[:16])
    return dct
