from __future__ import division
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from load_model import model
from util.utils import img2tensor, unitization, reduce_normalize
from multi_task.min_norm_solvers import  MinNormSolver
def attack(model,ori, outputs,ith):
    adv = ori.cuda().detach()
    adv.requires_grad = True
    divid_factor=0.15
    epochches=200
   
    
    y = unitization(model(reduce_normalize(ori.clone())))
    y2_label= torch.mm(y,torch.Tensor(outputs).t().cuda()).topk(2)[1].cpu().numpy()[0,1]
    y2=torch.Tensor(outputs[y2_label]).cuda()

    delta_dir = '0000' + str(y2_label)#loading second-rank  as adversarial perturbtion, marked as delta
    delta_dir = delta_dir[-5:] + '.jpg'
    delta = Image.open(os.path.join(ori_images_dir, delta_dir))
    delta = np.asarray(delta, np.float32)
    delta = delta / 255 * divid_factor
    delta = np.transpose(delta, (2, 0, 1))
    delta = torch.from_numpy(delta)
    delta = delta.unsqueeze(0)
    delta = Variable(delta.cuda(), requires_grad=True)
    
    optimizer = optim.Adam([delta], lr=4e-3, weight_decay=1e-4)
    for epoch in range(epochches):
        adv_data = adv.clone() + delta
        y1 = unitization(model(reduce_normalize(adv_data)))
        loss1 = torch.dot(y.reshape(-1),y1.reshape(-1))#force adv_data marked
                                                       #least-likely label
                                                       #compared to original example
        
        loss2 = -torch.dot(y1.reshape(-1), y2.reshape(-1))#force adv_data marked the same label to the image most
        
        loss3 = (torch.abs(torch.norm(delta, 2) - 1) + torch.norm(delta, 2) + 1) / 2#constrain 2-norm of delta
    
        ones = Variable(torch.ones(1, 3, 112, 112).cuda())# constrain adv_data to interval [0,1]
        loss4 = F.relu(adv_data - ones) + F.relu(-adv_data)
        loss4 = torch.norm(loss4, 2)
    
        loss1 = (torch.abs(loss1 + 0.3) + (loss1 - 0.3)) / 2 # constrain attack intensity
        # loss2 = (torch.abs(loss1 + 0.5) + (loss2 - 0.5)) / 2
        # employ multi-task learning method to weight losses, according to https://arxiv.org/abs/1810.04650
        loss_data = {}
        grads = {}
        scale = {}
        loss_list = [100*loss1,40*loss2, 246*loss3]
        for t in range(3):
            optimizer.zero_grad()
            loss_data[t] = loss_list[t].data
            loss_list[t].backward(retain_graph=True)
            grads[t] = Variable(delta.grad.data.clone().view(3,-1), requires_grad=False)
        gn = gradient_normalizers(grads, loss_data, 'l2')

        tasks = [0, 1, 2]
        for t in tasks:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / (float(gn[t])+1e-8)
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])

        for i, t in enumerate(tasks):
            scale[t] = float(sol[i])
        for i in range(3):
            if i == 0:
                loss = scale[i] * loss_list[i]
            else:
                loss = loss + scale[i] * loss_list[i]

        #loss = 100 * loss1 - 40 * loss2 + 45.2 * loss3
        # print(loss3.data.cpu().numpy())
        print('{}_the image:{}_th: {:.4f} | attack performance {:.4f} | {}|{}'. \
              format(ith, epoch, float(loss1.cpu().data.numpy()), float(loss2.cpu().data.numpy()),
                     float(loss3.cpu().data.numpy()), float(loss4.cpu().data.numpy())))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    adv_data = adv.clone() + delta
    return adv_data.detach()


def outputofbackbone(file):# calculate normlized outputs of testing images w.r.t backbone model
    outputs = []
    if not os.path.exists(adv_img_dir):
        os.mkdir(adv_img_dir)
    for id, img, _ in file:
        img = Image.open(os.path.join(ori_images_dir, img))
        tensor = img2tensor(img)
        output = unitization(model(reduce_normalize(tensor.clone()))).detach().cpu().numpy()[0]
        outputs.append(output)

    outputs = np.array(outputs)
    np.save(outputs_dir, outputs)
    return outputs


if __name__=='__main__':
    ori_images_dir = 'ori_images'
    adv_img_dir = 'images'
    outputs_dir = './data/outputs.pkl'
    file = pd.read_csv('./data/securityAI_round1_dev.csv')
    file = np.array(file)
    outputs=outputofbackbone(file)
    
    
    for i, img, _ in file:
        ith=i+1
        ori_img = Image.open(os.path.join(ori_images_dir, img))
        ori_tensor = img2tensor(ori_img)
        adv = attack(model=model, ori=ori_tensor, outputs=outputs, ith=i)
        adv_arr = adv.cpu().numpy().squeeze().transpose(1, 2, 0) * 255
        adv_arr = adv_arr[:, :, ::-1]
        cv2.imwrite(os.path.join(adv_img_dir, img), adv_arr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
