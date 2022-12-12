import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models import densenet161, DenseNet161_Weights
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from CloudDataset import CloudDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
weights = DenseNet161_Weights.DEFAULT
preprocess = weights.transforms()

model = densenet161(weights=weights).to(device)
print(model)

for i, param in enumerate(model.parameters()):
    print(i)
