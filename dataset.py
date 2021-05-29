import torch
import os
import glob
import cv2
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.images = []
        self.masks = []

        image_path = glob.glob(data_path+'/images/*.png')
        mask_path = glob.glob(data_path+'/onlylabels/*.png')
        
        for path in image_path:
            self.images.append(path)
        for path in mask_path:
            self.masks.append(path)

        self.transformI = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transformM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0],[0.5]) 
        ])

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = cv2.imread(image_path)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
        image = self.transformI(image)
        mask = self.transformM(mask)
        mask = mask.numpy()
        for px in mask:
            px[px>0] = 1
        return image, torch.from_numpy(mask)

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    dataset = CustomDataset("./final/train/") # test thu voi tap train
    # khong anh huong khi import duoi dang 1 lib
    print(dataset.__getitem__(1)[1])
