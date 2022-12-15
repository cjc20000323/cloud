import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models import densenet161, DenseNet161_Weights, densenet121, DenseNet121_Weights, efficientnet_b7, EfficientNet_B7_Weights
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from CloudDataset import CloudDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm
from transformers import BeitModel

config = 'microsoft/beit-base-patch16-224-pt22k-ft22k'
model = BeitModel.from_pretrained(config)
print(model)