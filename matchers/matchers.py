import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.svm import OneClassSVM
from face.adversarial import adv_utils as aut

_allowed_modes = ["flat", "sigmoid"]


def get_matchers_list(mode):
    cnt = CentroidMatcher(mode=mode)
    mxm = MaximumMatcher(mode=mode)
    lsvm = OneClassSvmMatcher(mode=mode, nu=0.5, kernel="linear")
    psvm = OneClassSvmMatcher(mode=mode, nu=0.5, kernel="poly")
    rsvm = OneClassSvmMatcher(mode=mode, nu=0.5, kernel="rbf")
    return [cnt, mxm, lsvm] # psvm, rsvm]


def shifted_sigm(x, a=5.0):
    """
    :param a:
    :return: computes 1/(1+e^-(x-a))
    """
    return 1.0/(1.0+np.e**-(x-a))


def get_weights(n_samples, mode):
    assert mode in _allowed_modes
    if mode == "flat":
        return np.ones(n_samples)/n_samples
    if mode == "sigmoid":
        max_x = 10.0
        scale = max_x/(n_samples-1)
        half = max_x/2
        xs = np.arange(0, (n_samples-.99), 1) * scale
        ys = np.array([shifted_sigm(i, half) for i in xs])
        return ys


class BaseMatcher(object):
    def __init__(self, **kwargs):
        pass

    def fit(self, traindata, **kwargs):
        raise RuntimeError("Fit not implemented")

    def predict(self, testdata, **kwargs):
        raise RuntimeError("predict not implemented")


def check_form(data):
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    assert data.shape[0] > 0


class CentroidMatcher(BaseMatcher):

    def __init__(self, mode, **kwargs):
        super(BaseMatcher).__init__()
        assert mode in _allowed_modes
        self.mode = mode
        self.class_name = "cnt"

    def get_name(self):
        return "cnt_" + self.mode

    def fit(self, traindata, **kwargs):
        check_form(traindata)
        self.template = np.array(traindata)
        self.weights = get_weights(traindata.shape[0], mode=self.mode)

    def predict(self, testdata, **kwargs):
        check_form(testdata)
        centroid = np.average(self.template, axis=0, weights=self.weights)
        d = aut.elem_wise_l2(testdata, centroid)
        return np.array(d)*-1


class MaximumMatcher(BaseMatcher):

    def __init__(self, mode, **kwargs):
        super(BaseMatcher).__init__()
        assert mode in _allowed_modes
        self.mode = mode

    def get_name(self):
        return "mxm_" + self.mode

    def fit(self, traindata, **kwargs):
        check_form(traindata)
        self.template = np.array(traindata)
        self.weights = get_weights(traindata.shape[0], mode=self.mode)

    def predict(self, testdata, **kwargs):
        check_form(testdata)
        d = []
        for i in range(testdata.shape[0]):
            d_here = aut.elem_wise_l2(self.template, testdata[i]) * (1-self.weights)
            d.append(d_here.min())
        return np.array(d)*-1


class OneClassSvmMatcher(BaseMatcher):

    def __init__(self, mode, nu, kernel, gamma="auto", **kwargs):
        super(BaseMatcher).__init__()
        self.svm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self._class_name = kernel[0] + "svm"
        assert mode in _allowed_modes
        self.mode = mode

    def get_name(self):
        return self._class_name + "_" + self.mode

    def fit(self, traindata, **kwargs):
        check_form(traindata)
        self.weights = get_weights(traindata.shape[0], mode=self.mode)
        self.svm.fit(traindata, sample_weight=self.weights)

    def predict(self, testdata, **kwargs):
        check_form(testdata)
        return self.svm.decision_function(testdata)
