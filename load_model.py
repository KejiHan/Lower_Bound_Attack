from PIL import Image
import torch
import torch.nn as nn

import os

from backbone.model_irse import IR_50
cpt_dir='./model/backbone_ir50_ms1m.pth'
IMAGE_SIZE=[112,112]
model =IR_50(IMAGE_SIZE)
model.load_state_dict(torch.load(cpt_dir))
model = model.cuda()
model = model.eval()