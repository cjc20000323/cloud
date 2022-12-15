import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision.models import densenet161, DenseNet161_Weights, densenet121, DenseNet121_Weights, resnet101, \
    ResNet101_Weights, resnet152, ResNet152_Weights, efficientnet_b0, EfficientNet_B0_Weights
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from CloudDataset import CloudDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm.auto import tqdm

checkpoint = 'efficientnetB2.pth'
batch_size = 32
img_path = 'cloud_dataset/images'
test_csv = 'cloud_dataset/test.csv'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
preprocess = EfficientNet_B0_Weights.DEFAULT.transforms()

img_transform = transforms.Compose([
    transforms.Resize(232),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
])


def test(model, dataloader):
    with torch.no_grad():
        y_pred = []
        filenames = []
        for filename, X, _ in tqdm(dataloader):
            filenames = filenames + list(filename)
            X = X.to(device)
            out = model(X).cpu()
            y_pred.append(out.argmax(dim=1) + 1)
    y_pred = torch.cat(y_pred)
    print(filenames)
    return filenames, y_pred


if __name__ == '__main__':
    test_df = pd.read_csv(test_csv)
    test_dataset = CloudDataset(test_df, img_path, preprocess, img_transform, True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = efficientnet_b0().to(device)
    # model.classifier = nn.Linear(in_features=1024, out_features=28, device=device)
    # model.fc = nn.Linear(in_features=2048, out_features=28, device=device)
    model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                     nn.Linear(in_features=1280, out_features=28, bias=True, device=device))
    model.load_state_dict(torch.load(checkpoint))

    model.eval()
    filenames, y_pred = test(model, test_dataloader)
    result = pd.DataFrame({'FileName': filenames, 'Code': y_pred})
    result.to_csv('densenet_baseline.csv', index=False)
    result.to_csv('submission.csv', index=False)
