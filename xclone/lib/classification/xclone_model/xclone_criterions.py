import numpy as np

def accuracy_criterion(xclone_instance):
    acc_history = xclone_instance._history["accuracy"]
    return acc_history[-1] == np.max(acc_history)


def loglik_criterion(xclone_instance):
    negloglik_history = xclone_instance._history["negloglik"]
    return negloglik_history[-1] == np.min(negloglik_history)
