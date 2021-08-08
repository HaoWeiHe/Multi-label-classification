import torch
import torch.nn as nn
from violation.ModelDefine import MultiLabel
from violation.load_bert import ViolationModel

model = ViolationModel()
predict = model.demo
