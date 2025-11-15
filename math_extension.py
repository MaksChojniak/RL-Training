import numpy as np

def ReLu(x:float) -> float:
    return np.max([0, x])

def Sigmoid(x:float) -> float:
    return 1 / (1 + np.pow(np.e, -x))