from torch.utils.data import Dataset
import os
from torchvision.io import read_image, ImageReadMode


class CloudDataset(Dataset):
    def __init__(self, df, img_path, preprocess, img_transform, test=False):
        super(CloudDataset).__init__()
        df.reset_index(inplace=True)
        self.filename = df['FileName']
        if not test:
            self.labels = df['Code'] - 1
        self.img_path = img_path
        self.preprocess = preprocess
        self.img_transform = img_transform
        self.test = test

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, item):
        filename = self.filename[item]
        img = read_image(os.path.join(self.img_path, self.filename[item]), mode=ImageReadMode.RGB)
        if not self.test:
            img = self.img_transform(img)
        img = self.preprocess(img)
        if not self.test:
            label = self.labels[item]
        else:
            label = -1
        return filename, img, label
