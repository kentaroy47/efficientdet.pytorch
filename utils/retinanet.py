'''RetinaFPN in PyTorch.
See the paper "Focal Loss for Dense Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from utils.ssd_model import DBox, Detect

class Bottleneck(nn.Module):
    """
    Resnet bottleneck blocks.
    res18 [2, 2, 2, 2]
    res34 [3, 4, 6, 3]
    res50 [3, 4, 6, 3]
    res101[3, 4, 23, 3]
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):
    loc_layers = []
    conf_layers = []

    # VGGの22層目、conv4_3（source1）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[0]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[0]
                              * num_classes, kernel_size=3, padding=1)]

    # VGGの最終層（source2）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[1]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[1]
                              * num_classes, kernel_size=3, padding=1)]

    # extraの（source3）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[2]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[2]
                              * num_classes, kernel_size=3, padding=1)]

    # extraの（source4）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                              * num_classes, kernel_size=3, padding=1)]

    # extraの（source5）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                              * num_classes, kernel_size=3, padding=1)]

    # extraの（source6）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                              * num_classes, kernel_size=3, padding=1)]
    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

class RetinaFPN(nn.Module):
    def __init__(self, block, num_blocks, phase, cfg, verbose=False, model="resnet18"):
        super(RetinaFPN, self).__init__()
        self.in_planes = 64
        
        # meta-stuff
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        self.verbose = verbose
        
        # make Dbox
        dbox = DBox(cfg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dbox_list = dbox.make_dbox_list()
        
        # use Detect if inference
        if phase == "inference":
            self.detect = Detect()
        
        import torchvision.models
        # define resnet18
        if model == "resnet18":       
            resnet=torchvision.models.resnet18(pretrained=True)     
            #print(resnet)
            ratio = 1
        elif model == "resnet34":
            resnet=torchvision.models.resnet34(pretrained=True)     
            #print(resnet)
            ratio = 4
        elif model == "resnet50":
            resnet=torchvision.models.resnet50(pretrained=True)     
            #print(resnet)
            ratio = 4
        elif model == "resnet101":
            resnet=torchvision.models.resnet101(pretrained=True)     
            #print(resnet)
            ratio = 4
        elif model == "resnet152":
            resnet=torchvision.models.resnet152(pretrained=True)     
            #print(resnet)
            ratio = 4
            
        ## CNN layers
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer2 = nn.Sequential(resnet.layer1)
        self.layer3 = nn.Sequential(resnet.layer2)
        self.layer4 = nn.Sequential(resnet.layer3)
        self.layer5 = nn.Sequential(resnet.layer4)

        # Bottom-up layers
        self.conv6 = nn.Conv2d( 512*ratio, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d( 256, 256, kernel_size=3, stride=1, padding=0)
        # Top layer
        self.toplayer = nn.Conv2d(512*ratio, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)        
        # Lateral layers
        self.latlayer1 = nn.Conv2d( 256*ratio, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 128*ratio, 256, kernel_size=1, stride=1, padding=0)     
        # loc, conf layers
        self.loc, self.conf = make_loc_conf(self.num_classes, cfg["bbox_aspect_num"])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = self.layer0(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4) 
        p6 = self.conv6(c5) # 5x5
        p7 = self.conv7(F.relu(p6)) # 3x3
        p8 = self.conv8(F.relu(p7)) # 1x1
        # Top-down
        p5 = self.toplayer(c5) # 10x10
        p4 = self._upsample_add(p5, self.latlayer1(c4)) # 19x19
        p3 = self._upsample_add(p4, self.latlayer2(c3)) # 38x38
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        
        # make loc and confs.
        sources = [p3, p4, p5, p6, p7, p8]
        
        # look at source size
        if self.verbose:
            for source in sources:
                print("layer size:", source.size())
        
        # make lists
        loc = list()
        conf = list()
        
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # Permuteは要素の順番を入れ替え
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        # locとconfの形を変形
        # locのサイズは、torch.Size([batch_num, 34928])
        # confのサイズはtorch.Size([batch_num, 183372])になる
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        # さらにlocとconfの形を整える
        # locのサイズは、torch.Size([batch_num, 8732, 4])
        # confのサイズは、torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        # これで後段の処理につっこめるかたちになる。
        
        output = (loc, conf, self.dbox_list)
        
        if self.phase == "inference":
            # Detectのforward
            return self.detect(output[0], output[1], output[2].to(self.device))
        else:
            return output


def RetinaFPN101():
    # return RetinaFPN(Bottleneck, [2,4,23,3])
    return RetinaFPN(Bottleneck, [2,2,2,2])


def test():
    net = RetinaFPN101()
    fms = net(Variable(torch.randn(1,3,600,900)))
    for fm in fms:
        print(fm.size())