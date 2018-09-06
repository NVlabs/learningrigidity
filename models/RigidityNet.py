import torch
import torch.nn as nn
import torchvision.models as models

from torch.autograd import Variable
from torch.nn import init

def conv(inplanes, outplanes, ks=3, st=1):
    return nn.Sequential(
        nn.Conv2d( inplanes, outplanes, kernel_size=ks, stride=st, padding=(ks-1)//2, bias=True),
        nn.BatchNorm2d(outplanes),
        nn.ReLU(inplace=True)
    )

def transpose_conv(inplanes, outplanes, ks=4, st=2):
    return nn.Sequential(
        nn.ConvTranspose2d(inplanes, outplanes, kernel_size=ks, stride=st, padding=(ks-1)//2, bias=True),
        nn.ReLU(inplace=True)
    )

class RigidityNet(nn.Module):
    """ The Rigidity Transform network 
    """
    def __init__(self):
        super(RigidityNet, self).__init__()

        self.conv_ch = [12, 32, 64, 128, 256, 512, 1024]

        self.conv1 = conv(self.conv_ch[0], self.conv_ch[1], 7, 2)
        self.conv2 = conv(self.conv_ch[1], self.conv_ch[2], 7, 2)
        self.conv3 = conv(self.conv_ch[2], self.conv_ch[3], 5, 2)
        self.conv4 = conv(self.conv_ch[3], self.conv_ch[4], 3, 2)
        self.conv5 = conv(self.conv_ch[4], self.conv_ch[5], 3, 2)
        self.conv6 = conv(self.conv_ch[5], self.conv_ch[6], 3, 1)

        self.predict_translate = nn.Conv2d(1024, 3, kernel_size=1, stride=1)
        self.predict_rotate = nn.Conv2d(1024, 3, kernel_size=1, stride=1)

        self.transpose_conv_ch = [32, 64, 128, 256, 512, 1024]

        self.transpose_conv5 = transpose_conv(self.transpose_conv_ch[5], self.transpose_conv_ch[4])
        self.transpose_conv4 = transpose_conv(self.transpose_conv_ch[4], self.transpose_conv_ch[3])
        self.transpose_conv3 = transpose_conv(self.transpose_conv_ch[3], self.transpose_conv_ch[2])
        self.transpose_conv2 = transpose_conv(self.transpose_conv_ch[2], self.transpose_conv_ch[1])
        self.transpose_conv1 = transpose_conv(self.transpose_conv_ch[1], self.transpose_conv_ch[0])

        # Use 1x1 convolution to predict the final mask
        self.predict_fg5 = nn.Conv2d(self.transpose_conv_ch[4], 2, kernel_size=1, stride=1)
        self.predict_fg4 = nn.Conv2d(self.transpose_conv_ch[3], 2, kernel_size=1, stride=1)
        self.predict_fg3 = nn.Conv2d(self.transpose_conv_ch[2], 2, kernel_size=1, stride=1)
        self.predict_fg2 = nn.Conv2d(self.transpose_conv_ch[1], 2, kernel_size=1, stride=1)
        self.predict_fg1 = nn.Conv2d(self.transpose_conv_ch[0], 2, kernel_size=1, stride=1)

        self._initialize_weights()

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        bottleneck = self.conv6(out_conv5)

        t = self.predict_translate(bottleneck)
        R = self.predict_rotate(bottleneck)

        out_transpose_conv5 = self.transpose_conv5(bottleneck)
        out_transpose_conv4 = self.transpose_conv4(out_transpose_conv5)
        out_transpose_conv3 = self.transpose_conv3(out_transpose_conv4)
        out_transpose_conv2 = self.transpose_conv2(out_transpose_conv3)
        out_transpose_conv1 = self.transpose_conv1(out_transpose_conv2)

        rg5 = self.predict_fg5(out_transpose_conv5)
        rg4 = self.predict_fg4(out_transpose_conv4)
        rg3 = self.predict_fg3(out_transpose_conv3)
        rg2 = self.predict_fg2(out_transpose_conv2)
        rg1 = self.predict_fg1(out_transpose_conv1)

        if self.training:
            return torch.cat([t,R], dim=1), rg1, rg2, rg3, rg4, rg5
        else:
            return torch.cat([t,R], dim=1), rg1

    def _initialize_weights(self):

        for idx, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform(m.weight.data) # the MSRA initialization
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                init.kaiming_uniform(m.weight.data) # the MSRA initialization
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def rigidity_transform_net(checkpoint_path=None):
    """ Load the rigidity attention network
    """
    model = RigidityNet()
    if checkpoint_path is not None:
        data = torch.load(checkpoint_path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model

if __name__ == '__main__':

    net = rigidity_transform_net()
    print(net)
