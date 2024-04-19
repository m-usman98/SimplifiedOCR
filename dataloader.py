from PIL import Image
import torch
import pandas as pd
import os
import torchvision.transforms as transforms


class load_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform, IMG_HEIGHT, IMG_WIDTH):
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.df = pd.read_csv(os.path.join(dataset_path, 'labels.csv'), sep=',', engine='python', encoding="ISO-8859-1")
        self.images_folder = dataset_path
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.IMG_HEIGHT, self.IMG_WIDTH)),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.at[index, "filename"]
        label = self.df.at[index, "words"]
        image = Image.open(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label