import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset4test import CustomDataset
from model import UNet
# from tqdm import tqdm
import torchvision.utils as vutils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
testDataset = CustomDataset('./final/test/')
testLoader = DataLoader(testDataset, batch_size = 1)
# label = testDataset.images_name
model = UNet(3, 1).to(device)
model.load_state_dict(torch.load("/content/drive/MyDrive/AI-ML/Lane_Segment_PyTorch/checkpointLaneSegment3.pth"))

model.eval()
for image in tq.tqdm(testLoader):
    input = image[0].to(device)
    # print("")
    smt = str(image[-1])[2:-3]
    # print(smt)
    vutils.save_image(model(input),'/content/drive/MyDrive/AI-ML/Lane_Segment_PyTorch/test/{}'.format(smt),normalize=True)