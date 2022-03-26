# ERFNET full network definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
multi_gpu=True
BatchNorm = SynchronizedBatchNorm2d if multi_gpu else nn.BatchNorm2d

class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = BatchNorm(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = BatchNorm(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = BatchNorm(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(1,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.1, 1))  

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))

        #only for encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)
        cnt=0
        for layer in self.layers:
            output = layer(output)
            if(cnt==0):
                x=output
                #print(x.size())
            cnt+=1
            

        if predict:
            output = self.output_conv(output)

        return x,output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = BatchNorm(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input
        cnt=0
        for layer in self.layers:
            output = layer(output)
            if(cnt==2):
                x=output
                #print(x.size())
            cnt+=1
            
        output = self.output_conv(output)

        return x,output


class ERFNet(nn.Module):
    def __init__(self, num_classes, partial_bn=False, encoder=None):  #use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)
        self.input_mean = [103.939, 116.779, 123.68] # [0, 0, 0]
        self.input_std = [1, 1, 1]
        self._enable_pbn = partial_bn

        if partial_bn:
            self.partialBN(True)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(ERFNet, self).train(mode)
        if self._enable_pbn:
            print("Freezing BatchNorm2D.")
            for m in self.modules():
                if isinstance(m, SynchronizedBatchNorm2d):
                    m.eval()
                    # shutdown update in frozen mode
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        base_weight = []
        base_bias = []
        base_bn = []

        addtional_weight = []
        addtional_bias = []
        addtional_bn = []

        # print(self.modules())

        for m in self.encoder.modules(): # self.base_model.modules()
            if isinstance(m, nn.Conv2d):
                # print(1)
                ps = list(m.parameters())
                base_weight.append(ps[0])
                if len(ps) == 2:
                    base_bias.append(ps[1])
            elif isinstance(m, SynchronizedBatchNorm2d):
                # print(2)
                base_bn.extend(list(m.parameters()))

        for m in self.decoder.modules(): # self.base_model.modules()
            if isinstance(m, SynchronizedBatchNorm2d):
                # print(1)
                ps = list(m.parameters())
                base_weight.append(ps[0])
                if len(ps) == 2:
                    base_bias.append(ps[1])
            elif isinstance(m, SynchronizedBatchNorm2d):
                # print(2)
                base_bn.extend(list(m.parameters()))


        return [
            {
                'params': addtional_weight,
                'lr_mult': 10,
                'decay_mult': 1,
                'name': "addtional weight"
            },
            {
                'params': addtional_bias,
                'lr_mult': 20,
                'decay_mult': 1,
                'name': "addtional bias"
            },
            {
                'params': addtional_bn,
                'lr_mult': 10,
                'decay_mult': 0,
                'name': "addtional BN scale/shift"
            },
            {
                'params': base_weight,
                'lr_mult': 1,
                'decay_mult': 1,
                'name': "base weight"
            },
            {
                'params': base_bias,
                'lr_mult': 2,
                'decay_mult': 0,
                'name': "base bias"
            },
            {
                'params': base_bn,
                'lr_mult': 1,
                'decay_mult': 0,
                'name': "base BN scale/shift"
            },
        ]

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            middle_feature1,output = self.encoder(input)            #predict=False by default
            # mid_x = torch.sum(output, 1) # newly_added
            # mid_x = mid_x * mid_x
            middle_feature2,output=self.decoder.forward(output)
            return middle_feature1,middle_feature2,output #, output #, mid_x
            
            
if __name__ == '__main__':
    # Debug
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    #net = MobileNetV2_unet(pre_trained=None)
    #net(torch.randn(1, 3, 224, 224))
    batch_size = 8
    num_classes = 2
    h = 256
    w = 256
    x = torch.randn((batch_size,1, h, w), requires_grad=True)
    net = ERFNet(2)
    middle_feature1,middle_feature2,y = net(x)
    print(middle_feature1.size())
    print(middle_feature2.size())
    
    print(y.size())