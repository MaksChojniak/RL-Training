import numpy as np
from math_extension import ReLu, Sigmoid
import torch as t

print(t.__version__)
print("CUDA available:", t.cuda.is_available())
print("Device count:", t.cuda.device_count())
print("Current device:", t.cuda.current_device())
print("Device name:", t.cuda.get_device_name(0))
