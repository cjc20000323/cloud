import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models import densenet161, DenseNet161_Weights, densenet121, DenseNet121_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights, efficientnet_b0, EfficientNet_B0_Weights, resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from CloudDataset import CloudDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
import random
from loss import MultiClassFocalLossWithAlpha
from transformers import BeitModel

lr = 5e-4
epochs = 25
batch_size = 16
img_path = 'cloud_dataset/images'
train_csv = 'cloud_dataset/train.csv'
test_csv = 'cloud_dataset/test.csv'
aug_train_csv = 'cloud_dataset/new_train.csv'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

img_transform = transforms.RandomApply([
    transforms.Resize(400),
    transforms.RandomCrop(384),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.5, 0.5, 0.5)
], p=0.5)

writer = SummaryWriter()

freeze_layer_num = 2


def write_log(name, value, step):
    writer.add_scalar(name + '_accuracy', value[0], global_step=step)
    writer.add_scalar(name + '_recall', value[1], global_step=step)
    writer.add_scalar(name + '_f1', value[2], global_step=step)


def freeze(model, freeze_layer_num):
    for i, param in enumerate(model.parameters()):
        if i < 484 - freeze_layer_num * 6:
            param.requires_grad = False


def train(model, dataloader, criterion, optimizer, scheduler):
    progress_bar = tqdm(range(len(dataloader)))
    y_true, y_pred = [], []
    alpha = 1.0
    probility = random.randint(0, 10)
    for file_name, X, y in dataloader:
        X = X.to(device)
        if probility > 7:
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(X.size(0)).cuda()
            X, X_mix = X, X[index]
            y, y_mix = y, y[index]
            X = lam * X + (1 - lam) * X_mix
            out = model(X).cpu()
            loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y_mix)
        else:
            out = model(X).cpu()
            loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        progress_bar.update(1)

        y_true.append(y)
        y_pred.append(out.argmax(dim=1))
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred,
                                                                                                   average='macro')


def eval(model, dataloader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for _, X, y in dataloader:
            X = X.to(device)
            out = model(X).cpu()
            y_true.append(y)
            y_pred.append(out.argmax(dim=1))
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred,
                                                                                                   average='macro')


if __name__ == '__main__':
    # train_df = pd.read_csv(train_csv)
    aug_train_df = pd.read_csv(aug_train_csv)
    train_df, eval_df = train_test_split(aug_train_df, random_state=0, stratify=aug_train_df['Code'])
    train_dataset = CloudDataset(train_df, img_path, preprocess, img_transform)
    eval_dataset = CloudDataset(eval_df, img_path, preprocess, img_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    model = resnet50(weights=weights).to(device)
    print(model)
    # model.classifier = nn.Linear(in_features=2208, out_features=28, device=device)
    # model.heads.head = nn.Linear(in_features=1024, out_features=28, device=device)
    model.fc = nn.Linear(in_features=2048, out_features=28, device=device)
    # model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(in_features=1280, out_features=28, bias=True, device=device))

    '''
    for i, param in enumerate(model.parameters()):
        if i < 484 - freeze_layer_num * 2:
            param.requires_grad = False
    '''

    optimizer = Adam(model.parameters(), lr=lr)
    total_iters = epochs * len(train_dataloader)
    scheduler = LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=total_iters)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0
    best_state_dict = {}
    state_dict = {}
    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}')
        model.train()
        acc, rec, f1 = train(model, train_dataloader, criterion, optimizer, scheduler)
        print(f'Train Acc: {acc}, Rec: {rec}, F1 score: {f1}')
        write_log('train', [acc, rec, f1], epoch + 1)
        model.eval()
        acc, rec, f1 = eval(model, eval_dataloader)
        if f1 > best_f1:
            best_f1 = f1
            best_state_dict = model.state_dict()
        print(f'Eval Acc: {acc}, Rec: {rec}, F1 score: {f1}')
        write_log('eval', [acc, rec, f1], epoch + 1)
    torch.save(best_state_dict, 'resnet50.pth')
