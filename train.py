import numpy as np
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
from sklearn.metrics import f1_score, accuracy_score, recall_score
from tqdm.auto import tqdm

lr = 1e-5
epochs = 25
batch_size = 16
img_path = 'cloud_dataset/images'
train_csv = 'cloud_dataset/train.csv'
test_csv = 'cloud_dataset/test.csv'
aug_train_csv = 'cloud_dataset/new_train.csv'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
weights = DenseNet161_Weights.DEFAULT
preprocess = weights.transforms()

img_transform = transforms.RandomApply(nn.ModuleList[
    transforms.Resize(232),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.5, 0.5, 0.5)
], p=0.5)


def train(model, dataloader, criterion, optimizer, scheduler):
    progress_bar = tqdm(range(len(dataloader)))
    y_true, y_pred = [], []
    alpha = 1.0
    for file_name, X, y in dataloader:
        X = X.to(device)
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(X.size(0)).cuda()
        X, X_mix = X, X[index]
        y, y_mix = y, y[index]
        X = lam * X + (1 - lam) * X_mix
        out = model(X).cpu()
        loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y_mix)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        progress_bar.update(1)

        y_true.append(y)
        y_pred.append(out.argmax(dim=1))
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')


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
    return accuracy_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')


if __name__ == '__main__':
    # train_df = pd.read_csv(train_csv)
    aug_train_df = pd.read_csv(aug_train_csv)
    train_df, eval_df = train_test_split(aug_train_df, random_state=0, stratify=aug_train_df['Code'])
    train_dataset = CloudDataset(train_df, img_path, preprocess, img_transform)
    eval_dataset = CloudDataset(eval_df, img_path, preprocess, img_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    model = densenet161(weights=weights).to(device)
    model.classifier = nn.Linear(in_features=2208, out_features=28, device=device)
    print(model)
    # model.heads.head = nn.Linear(in_features=1024, out_features=28, device=device)

    optimizer = Adam(model.parameters(), lr=lr)
    total_iters = epochs * len(train_dataloader)
    scheduler = LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=total_iters)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0
    best_state_dict = {}
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        model.train()
        acc, rec, f1 = train(model, train_dataloader, criterion, optimizer, scheduler)
        print(f'Train Acc: {acc}, Rec: {rec}, F1 score: {f1}')
        model.eval()
        acc, rec, f1 = eval(model, eval_dataloader)
        if f1 > best_f1:
            best_f1 = f1
            best_state_dict = model.state_dict()
        print(f'Eval Acc: {acc}, Rec: {rec}, F1 score: {f1}')
    torch.save(best_state_dict, 'densenet161.pth')
