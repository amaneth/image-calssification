import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from data.dataloader import CustomDataSet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
from PIL import Image

from torch.utils.data import DataLoader,random_split
import torch.optim as optim

from torch.utils.data import DataLoader
from config import args
from models import resnet, CNN
from train import train
from utils import transform, plot

metadata = args.metadata
path=  args.path


data= CustomDataSet(metadata, path, transform= transform)
batch_size= args.bs
train_size= args.train_size
train_size= int(train_size*len(data))
val_size = len(data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', help='This is name of the model', required=True, type=str)
parser.add_argument('-n', '--num_epochs', help='This is the number of epochs', required=True, type=str)

main_args = vars(parser.parse_args())
num_epochs = int(main_args['num_epochs'])
model_name = main_args['model_name'].lower()
# print(model_name=='resnet')
if model_name=='resnet':
    model = resnet()
elif model_name=='cnn':
    model= CNN()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion= nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
model_trained, percent, val_loss, val_acc, train_loss, train_acc= train(model, criterion, train_loader, val_loader, optimizer, num_epochs, device)

plot(train_loss, val_loss, model_name)