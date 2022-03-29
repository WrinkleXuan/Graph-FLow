#from torchstat import stat
import torchvision.models as models
import torch
from networks.MobileNetV2_unet import MobileNetV2_unet
from networks.enet import ENet
from networks.erfnet import ERFNet
from networks.Unet_SKA2 import UnetSKA2 
from torchvision.models import resnet34
from thop import profile
from thop import clever_format
from torchstat import stat
#macs, params = clever_format([flops, params], "%.3f")
#model = DinkNet50()


#model=ENet(2)
#model = ERFNet(2)
model= MobileNetV2_unet(pre_trained=None)
#model =UnetSKA2(in_ch=1,out_ch=2)
###method 1
#stat(model, (1, 256, 256))


import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  net = ENet(2)
  #net=MobileNetV2_unet(pre_trained=None)
  macs, params = get_model_complexity_info(net, (1, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))



###method 2
"""
input = torch.randn(1, 1, 256, 256)
flops, params = profile(model, inputs=(input, ))
macs, params = clever_format([float(flops), float(params)], "%.3f")
print (macs)
print (params)

"""


"""
### medthod 3

from torchsummary import summary
#import torchvision.models as models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#vgg = models.vgg19().to(device)
model = DinkNet50().to(device)

#model = model.cuda()
summary(model, input_size=(3,32,32), batch_size=1 , device='cuda')
"""