import pandas as pd
import torchvision.transforms as transforms
from torchvision.io import read_image, ImageReadMode, write_png
import os

from tqdm import tqdm

images_path = 'cloud_dataset/images'
train_csv = 'cloud_dataset/train.csv'
new_train_csv = 'cloud_dataset/new_train.csv'

loc = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']


def write_image(img, img_path, df, code):
    write_png(img, images_path + '/' + img_path)
    df.loc[len(df)] = [img_path, code]


if __name__ == '__main__':
    train_df = pd.read_csv(train_csv, )
    print(len(train_df))
    new_train_df = train_df.copy(deep=True)
    for index, value in tqdm(train_df.iterrows()):
        if value['Code'] in [3, 6, 9, 12, 14]:
            img = read_image(os.path.join(images_path, value['FileName']), mode=ImageReadMode.RGB)
            randomHorizontalFlip = transforms.RandomHorizontalFlip(p=1)(img)
            write_image(randomHorizontalFlip, value['FileName'] + '_HorizontalFlip.png', new_train_df, value['Code'])
            randomVerticalFlip = transforms.RandomVerticalFlip(p=1)(img)
            write_image(randomVerticalFlip, value['FileName'] + '_VerticalFlip.png', new_train_df, value['Code'])
            randomRotation = transforms.RandomRotation(45)(img)
            write_image(randomRotation, value['FileName'] + '_Rotation.png', new_train_df, value['Code'])
            randomGray = transforms.Grayscale()(img)
            write_image(randomGray, value['FileName'] + '_Gray.png', new_train_df, value['Code'])
            img = transforms.Resize(1080)(img)
            TenCrop = transforms.TenCrop(720)(img)
            for index, data in enumerate(TenCrop):
                write_image(data, value['FileName'] + '_' + loc[index] + '.png', new_train_df, value['Code'])

    print(len(new_train_df))
    new_train_df.to_csv(new_train_csv, index=False)

