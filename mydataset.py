from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd


#
class MyDataSet(Dataset):
    """"""

    def __init__(self, images_path: list, gt_path: list, transform=None):
        self.images_path = images_path
        self.gt_path = gt_path
        self.transform = transform
        # self.img_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        
        gt = pd.read_csv(self.gt_path[item], header=None)
        gt = torch.tensor(gt.values, dtype=torch.float32)

        return img, gt
