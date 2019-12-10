import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def img2tensor(img):
    tf=transforms.Compose([
        transforms.ToTensor()
    ])
    tensor=tf(img)
    tensor = tensor.unsqueeze(0)
    tensor=tensor.cuda()
    
    return tensor



mean = [0.5, 0.5, 0.5]
std = [0.5,0.5, 0.5]
def reduce_normalize(tensor):# normalize values of each channel of the tensor
    chan_number=tensor.size()[1]
    for i in range(chan_number):
        tensor[:, i, :, :] = (tensor[:, i, :, :] - mean[i])/std[i]
    
    return tensor


def unitization(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    #output=input
    return output

if __name__=='__main__':
    img=Image.open('/home/hankeji/Desktop/cat.jpg')
    tensor=img2tensor(img,'cuda:0')
    tensor=tensor.unsqueeze(0)
    reduce_normalize(tensor)