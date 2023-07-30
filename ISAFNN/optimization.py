import os
import random
import datetime
import numpy
import math
import time
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

# os.environ['CUDA_VISIBLE_DEVICES']='1'

# NOTE: parameter setting
NET_LAYERS = 5  # network depth
NET_CHANNELS = 16  # network width
ref_images = ['Color_battery-128.bmp', 'Color_battery-128.bmp', 'Color_battery-128.bmp']
ref_size = 128  # image size
iterations = 1000
save_slice = 50
learning_rate = 0.1
batch_size = 1
random_seed = 0
random.seed(random_seed)
torch.manual_seed(random_seed)
initial_noise = torch.rand(1, 3, ref_size + 2 * NET_LAYERS, ref_size + 2 * NET_LAYERS,
                           ref_size + 2 * NET_LAYERS)


# NOTE: description function
class ACFloss(nn.Module):
    def __init__(self):
        super(ACFloss, self).__init__()

    def forward(self, x, t):
        x0 = torch.mean(x, dim=1)
        t0 = torch.mean(t, dim=1).squeeze().unsqueeze(0).unsqueeze(0)
        samp = x0.squeeze().unsqueeze(0).unsqueeze(0)
        k1 = samp
        k2 = t0
        flip_s = torch.flipud(samp)
        flip_s = torch.fliplr(flip_s)
        flip_t = torch.flipud(t0)
        flip_t = torch.fliplr(flip_t)
        out_x = F.conv2d(flip_s, k1, padding=int(ref_size / 2))
        out_t = F.conv2d(flip_t, k2, padding=int(ref_size / 2))
        #out_x = F.conv2d(flip_s, k1, padding=40)
        #out_t = F.conv2d(flip_t, k2, padding=40)
        loss = torch.abs(out_x - out_t).mean() / 100
        return loss


class VGG(nn.Module):
    def __init__(self, pool='max', pad=1):
        super(VGG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=pad)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=pad)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=pad)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=pad)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=pad)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=pad)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w * c)
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return (out)


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
    #print('img',tmp.mean())
    img = pro_b(tmp)
    return img


# NOTE: data process
def data_norm(in_data):
    d_min = in_data.min()
    if (d_min < 0):
        in_data += torch.abs(d_min)
        d_min = in_data.min()
    d_max = in_data.max()
    dst = d_max - d_min
    norm_data = (in_data - d_min).true_divide(dst)
    return norm_data


def calc_size_input(h, w, d, pad):
    s = [math.ceil(h + 2 * pad),
         math.ceil(w + 2 * pad),
         math.ceil(d + 2 * pad)]
    return s


# NOTE: network design
lmcn = LmCn(net_layers=NET_LAYERS, net_channels=NET_CHANNELS)
print(lmcn)  # display network structure
params = list(lmcn.parameters())
total_parameters = 0
for p in params:
    total_parameters = total_parameters + p.data.numpy().size
print('total number of parameters = ' + str(total_parameters))
lmcn.cuda()

# NOTE: description function
vgg = VGG(pool='avg', pad=1)
vgg.load_state_dict(torch.load('./Models/vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
vgg.cuda()
loss_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
loss_fns = [GramMSELoss()] * len(loss_layers)
loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
w = [1, 1, 1, 1, 1]
acf = ACFloss()
acf.cuda()

# NOTE: save filepath
time_info = datetime.datetime.now()
out_folder_name = time_info.strftime("%Y-%m-%d-%H%M") + '_' \
                  + ref_images[0][:-4] \
                  + '_Net_l' + str(NET_LAYERS) + 'c' + str(NET_CHANNELS)
if not os.path.exists('./Optimization/' + out_folder_name):
    os.mkdir('./Optimization/' + out_folder_name)

# NOTE: Optimization process
# load images and define target domain
directions = [0, 1, 2]
train_images = [Image.open('./RefImages/' + name) for name in ref_images]
train_images_torch = [Variable(img2tensor(img)).unsqueeze(0).cuda() for img in train_images]
targets = []
for img in train_images_torch:
    targets.append([GramMatrix()(f).detach() for f in vgg(img, loss_layers)])
target_domain = vgg(train_images_torch[0], loss_layers)

# determine the optimizer
optimizer = optim.Adam(lmcn.parameters(), lr=learning_rate)
loss_history = numpy.zeros(iterations)

# Optimization start
time_begin = time.time()
for n_iter in range(iterations):
    optimizer.zero_grad()

    # three directions
    for idx, d in enumerate(directions):
        target = targets[idx]
        output_sizes = [ref_size for N in range(3)]
        output_sizes[d] = 1
        random_point = [0, 0, 0]

        for i in range(batch_size):
            # random position
            random_point[d] = random.randint(0, ref_size - 1)
            #random_point[d] = random.randint(0, 3)*20
            # get input area
            input_size = calc_size_input(output_sizes[0], output_sizes[1], output_sizes[2], NET_LAYERS)
            input_area = initial_noise[:, :, random_point[0]:random_point[0] + input_size[0],
                         random_point[1]:random_point[1] + input_size[1],
                         random_point[2]:random_point[2] + input_size[2]]
            z_samples = Variable(input_area.cuda())

            # mapping
            rec_sample = lmcn(z_samples)
            if d == 0:
                sample = rec_sample[:, :, 0, 0:output_sizes[1]:, 0:output_sizes[2]]
            if d == 1:
                sample = rec_sample[:, :, 0:output_sizes[0], 0, 0:output_sizes[2]]
            if d == 2:
                sample = rec_sample[:, :, 0:output_sizes[0], 0:output_sizes[1], 0]
            sample = sample.squeeze().unsqueeze(0)

            # NOTE: description function select
            # # vgg loss
            rec_domain = vgg(sample, loss_layers)
            losses = [w[a] * loss_fns[a](f, target[a]) for a, f in enumerate(rec_domain)]
            single_loss = (1 / (batch_size * len(targets))) * sum(losses)
            # # vgg loss
            # # acf loss
            #losses = acf(sample, train_images_torch[0])
            #single_loss = (1 / (batch_size * len(targets))) * losses
            # # acf loss
            single_loss.backward(retain_graph=False)
            loss_history[n_iter] = loss_history[n_iter] + single_loss.item()
            del losses, single_loss
            del rec_sample, z_samples, input_area

        if n_iter % save_slice == (save_slice - 1):
            #print('mean',sample.data.cpu().squeeze().mean())
            #print('1',sample.data.cpu().squeeze())
            out_img = tensor2img(sample.data.cpu().squeeze())
            #print('2',out_img)
            out_img.save('./Optimization/' + out_folder_name + '/d'
                         + str(d) + '_iter_' + str(n_iter + 1) + '.jpg', "JPEG")
        del sample
    print('Iteration: %d, loss: %f' % (n_iter, loss_history[n_iter]))
    optimizer.step()

# Optimization end
time_end = time.time()
time = time_end - time_begin
print(time)

# save final model and loss history
torch.save(lmcn, './Optimization/' + out_folder_name + '/opt_model.py')
torch.save(lmcn.state_dict(), './Optimization/' + out_folder_name + '/params.pytorch')
txt_path = './Optimization/' + out_folder_name + '/loss.txt'
file_handle = open(txt_path, mode='w')
file_handle.write("time:" + str(time) + '\n')
file_handle.write("seed:" + str(random_seed) + '\n')
for n_iter in range(iterations):
    file_handle.write((str(n_iter) + ":" + str(loss_history[n_iter])) + '\n')

# once rec
rec_folder = './Optimization/data/once_rec'
rec_in = Variable(initial_noise, volatile=True).cuda()
with torch.no_grad():
    rec_out = lmcn(rec_in)
for i in range(0, ref_size, 1):
    print(i)
    slice = rec_out[:, :, :, :, i]
    slice = slice.squeeze().unsqueeze(0)
    out_img = tensor2img(slice.data.cpu().squeeze())
    out_img.save('./' + rec_folder + '/middle_slice_' + str(i) + '.bmp')
