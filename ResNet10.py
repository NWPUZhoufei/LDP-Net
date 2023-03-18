import torch
import torch.nn as nn
import math


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class SimpleBlock(nn.Module):
    def __init__(self, indim, outdim, stride, dilated_rate):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.stride = stride
        self.dilated_rate = dilated_rate
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=self.stride, dilation=self.dilated_rate, padding=self.dilated_rate, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]
        if indim!=outdim:
            self.shortcut = nn.Conv2d(indim, outdim, kernel_size=1, stride=self.stride, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)
            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'
        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

 
class ResNet(nn.Module):
    def __init__(self, list_of_out_dims=[64,128,256,512], list_of_stride=[1,2,2,2], list_of_dilated_rate=[1,1,1,1]):
        self.block = SimpleBlock
        super(ResNet,self).__init__()
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        init_layer(conv1)
        init_layer(bn1)
        trunk = [conv1, bn1, relu, pool1] 
        indim = 64 
        for i in range(4):
            B = self.block(indim, list_of_out_dims[i], list_of_stride[i], list_of_dilated_rate[i])
            trunk.append(B) 
            indim = list_of_out_dims[i]
        self.trunk = nn.Sequential(*trunk)
    def forward(self,x):
        out = torch.mean(self.trunk(x), [2,3]) # (B,512,7,7) 
        return out

def ResNet10():
    return ResNet(list_of_out_dims=[64,128,256,512], list_of_stride=[1,2,2,2], list_of_dilated_rate=[1,1,1,1])

if __name__=='__main__':
    
    model = ResNet10()
    x = torch.rand(1,3,224,224)
    y = model(x) # （5，512）
    print(y.size())



   