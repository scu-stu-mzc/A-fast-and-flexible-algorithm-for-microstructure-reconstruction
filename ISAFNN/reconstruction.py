import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import time
#import example
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# NOTE: parameter setting
rec_H = 512
rec_W = 512
rec_L = 512
piece_h = 128
piece_w = 128
piece_l = 128
NET_LAYERS = 8  # network depth
NET_CHANNELS = 16  # network width
ref_img_path = './RefImages/Color_0017.bmp'
model_folder = './Optimization/data/once_rec'
random_seed = 0
random.seed(random_seed)
torch.manual_seed(random_seed)
rec_noise = torch.rand(1, 3, rec_H + 2 * NET_LAYERS, rec_W + 2 * NET_LAYERS,
                       rec_L + 2 * NET_LAYERS)

# NOTE: network structure
class Conv3_Block(nn.Module):
    def __init__(self, input_channels, output_channels, m=0.1):
        super(Conv3_Block, self).__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, 3, padding=0, bias=True)
        self.bn = nn.BatchNorm3d(output_channels, momentum=m)

    def forward(self, x):
        x = F.leaky_relu(self.bn(self.conv(x)))
        return x


class LmCn(nn.Module):
    def __init__(self, net_layers=5, net_channels=16):
        super(LmCn, self).__init__()
        self.convs = []
        conv_layer_1 = Conv3_Block(3, net_channels)
        setattr(self, 'cb1_1', conv_layer_1)
        self.convs.append(conv_layer_1)
        for i in range(net_layers - 1):
            conv_layer_n = Conv3_Block(net_channels, net_channels)
            setattr(self, 'cb%i_1' % (i + 2), conv_layer_n)
            self.convs.append(conv_layer_n)
        last_conv = nn.Conv3d(net_channels, 3, 1, padding=0, bias=True)
        setattr(self, 'last_conv', last_conv)
        self.convs.append(last_conv)

    def forward(self, z):
        y = z
        for i in range(NET_LAYERS + 1):
            y = self.convs[i](y)
        return y


# NOTE: data type conversion
img2tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul_(255)),
])

pro_a = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255))])

pro_b = transforms.Compose([transforms.ToPILImage()])


def tensor2img(tensor):
    tmp = pro_a(tensor)
    tmp[tmp > 1] = 1
    tmp[tmp < 0] = 0
    img = pro_b(tmp)
    return img


# NOTE: reconstruction process
time_begin = time.time()
lmcn = LmCn(net_layers=NET_LAYERS, net_channels=NET_CHANNELS)
lmcn.load_state_dict(torch.load('./' + model_folder + '/params.pytorch'))
lmcn.cuda()

padding = 2 * NET_LAYERS
rec_out = torch.zeros(1, 3, rec_H, rec_W, rec_L)
with torch.no_grad():
    for i in range(0, rec_H, piece_h):
        for j in range(0, rec_W, piece_w):
            for k in range(0, rec_L, piece_l):
                tmp = rec_noise[:, :, i:i + piece_h + padding, j:j + piece_w + padding, k:k + piece_l + padding]
                area_in = Variable(tmp, volatile=True).cuda()
                area_out = lmcn(area_in)
                rec_out[:, :, i:i + piece_h, j:j + piece_w, k:k + piece_l] = area_out
                del tmp, area_in, area_out

time_end = time.time()
time = time_end - time_begin
print(time)
txt_path = './' + model_folder + '/rec_time.txt'
file_handle = open(txt_path, mode='w')
file_handle.write("time:" + str(time) + '\n')
for i in range(0, rec_L, 1):
    print(i)
    slice = rec_out[:, :, :, :, i]
    slice = slice.squeeze().unsqueeze(0)  # 1,3,128,128,1
    out_img = tensor2img(slice.data.cpu().squeeze())
    out_img.save('./' + model_folder + '/gray_slice_' + str(i) + '.bmp')

# NOTE: 3D segment process: gray2binary
"""
rec_out[rec_out > 1] = 1
rec_out[rec_out < 0] = 0
gt = example.GT()
gt.set(0, 0.5)
ref_img = Image.open(ref_img_path)
np_img = np.array(ref_img)
img_mean = np_img.mean() / 255
img_mean = img_mean.astype('float32')
print('ref image porosity:', img_mean)

gray_img3d = torch.mean(rec_out, dim=1).squeeze()
np_gray_img3d = gray_img3d.numpy()
data3d = np_gray_img3d.astype('float32')
data3d.tolist()
b = gt.img3dseg(data3d, img_mean)
b_numpy = np.array(b)
b_numpy = (b_numpy * 255).astype('uint8')
for i in range(0, rec_L, 1):
    print(i)
    slice = b_numpy[:, :, i]
    slice = slice.squeeze()
    out_img = Image.fromarray(slice)
    out_img.save('./' + model_folder + '/binary_slice_' + str(i) + '.bmp')
"""