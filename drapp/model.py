from functools import reduce
from drapp import pd

import sys
package_dir = 'resources/pretrained-models.pytorch-master/'
sys.path.insert(0, package_dir)
import numpy as np
import pandas as pd
import torchvision
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import pretrainedmodels
import math
import csv
from PIL import ExifTags, ImageStat
from PIL.ExifTags import TAGS, GPSTAGS

def submitDetails(filename):

    device = torch.device("cpu")
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    class RetinopathyDatasetTest(Dataset):
        def __init__(self, transform):
            self.transform = transform

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            img_name = os.path.join('drapp/static/', filename)
            image = Image.open(img_name)
            stat = ImageStat.Stat(image)
            r,g,b = stat.mean
            brightness = math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
            width, height = image.size
            img_format = str(image.format)
            num_pixels = stat.count[0]
            r_band, g_band, b_band = stat.extrema
            sum_of_pixels = stat.sum


            csv_columns = ['Name','Red','Green','Blue','Brightness','Width','Height','Format','Num_Pixels','R_Band','G_Band','B_Band','Sum_Pixels']

            dict_data = [{'Name':img_name,'Red': r, 'Green': g, 'Blue': b,'Brightness':brightness,'Width':width,'Height':height,'Format':img_format,'Num_Pixels':num_pixels,'R_Band':r_band,'G_Band':g_band,'B_Band':b_band,'Sum_Pixels':sum_of_pixels  },]


            csv_file="Image_Metadata.csv"

            try:
                with open(csv_file, 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for data in dict_data:
                        writer.writerow(data)
            except IOError:
                print("I/O error") 
            image = self.transform(image)
            return {'image': image}


    model = pretrainedmodels.__dict__['resnet101'](pretrained=None)

    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                              nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.25),
                              nn.Linear(in_features=2048, out_features=2048, bias=True),
                              nn.ReLU(),
                              nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Dropout(p=0.5),
                              nn.Linear(in_features=2048, out_features=1, bias=True),
                             )
    model.load_state_dict(torch.load("resources/model.bin" , map_location=torch.device('cpu')))
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = RetinopathyDatasetTest(transform=test_transform)

    test_data_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=0)
    test_preds = np.zeros((len(test_dataset), 1))
    tk0 = tqdm(test_data_loader)

    for i, x_batch in enumerate(tk0):
        x_batch = x_batch["image"]
        pred = model(x_batch.to(device))
        test_preds[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

    coef = [0.5, 1.5, 2.5, 3.5]

    for i, pred in enumerate(test_preds):
        if pred < coef[0]:
            test_preds[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            test_preds[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            test_preds[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            test_preds[i] = 3
        else:
            test_preds[i] = 4

    prslt = test_preds[0][0].astype(int)

    result_dict={
            "predclass": prslt
        }

    return result_dict

