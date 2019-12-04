from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.ssd_model import DBox, Detect
from BiFPN import BiFPN


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

class EfficientDet(nn.Module):
    def __init__(self, phase, cfg, verbose=False, backbone="efficientnet-b0", useBiFPN=True):
        super(EfficientDet, self).__init__()
        # meta-stuff
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        self.verbose=verbose
        # make Dbox
        dbox = DBox(cfg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dbox_list = dbox.make_dbox_list()        
        # use Detect if inference
        if phase == "inference":
            self.detect = Detect()
        ratio = 1
        
        # define backbone
        model = EfficientNet.from_pretrained(backbone)
        self.layer0 = nn.Sequential(model._conv_stem, model._bn0)
        self.layer2 = nn.Sequential(model._blocks[0],model._blocks[1],model._blocks[2],model._blocks[3])
        self.layer3 = nn.Sequential(model._blocks[4],model._blocks[5])
        self.layer4 = nn.Sequential(model._blocks[6],model._blocks[7],model._blocks[8],model._blocks[9],model._blocks[10],model._blocks[11])
        self.layer5 = nn.Sequential(model._blocks[12],model._blocks[13],model._blocks[14],model._blocks[15])
        # Bottom-up layers
        #self.conv5 = nn.Conv2d( 320, 256, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d( 320, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d( 256, 256, kernel_size=3, stride=1, padding=0)
        # Top layer
        self.toplayer = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)        
        # Lateral layers
        self.latlayer1 = nn.Conv2d( 80, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 40, 256, kernel_size=1, stride=1, padding=0)
        # loc, conf layers
        self.loc, self.conf = make_loc_conf(self.num_classes, cfg["bbox_aspect_num"])
        # FPNs
        self.usebifpn=useBiFPN
        if useBiFPN:
            self.BiFPN1=BiFPN(256)
            self.BiFPN2=BiFPN(256)
            print("use BiFPN")
        else:
            print("use FPN")
        
    def forward(self, x):
        ######### efficientnet layers ############
        x = self.layer0(x)
        p3 = self.layer2(x) # 37x37       
        p4 = self.layer3(p3) # 18x18       
        p5 = self.layer4(p4)
        p5 = self.layer5(p5)
        
        if self.verbose:
            print("layerc3:", p3.size())
            print("layerc4:", p4.size())
            print("layerc5:", p5.size())
            
        ######## non-efficientnet layers ###########
        p6 = self.conv6(p5) # 5x5
        p7 = self.conv7(F.relu(p6)) # 3x3
        p8 = self.conv8(F.relu(p7)) # 1x1
        
        ########### implement BiFPN ############
        if not self.usebifpn:
            # use FPN
            # Top-down
            p5 = self.toplayer(p5) # 10x10
            p4 = self._upsample_add(p5, self.latlayer1(p4)) # 19x19
            p3 = self._upsample_add(p4, self.latlayer2(p3)) # 38x38
            # Smooth
            p4 = self.smooth1(p4)
            p3 = self.smooth2(p3)
            # make loc and confs.
            sources = [p3, p4, p5, p6, p7, p8]
        else:
            # use BiFPNs
            # Top-down
            p5 = self.toplayer(p5) # 10x10
            p4 = self._upsample_add(p5, self.latlayer1(p4)) # 19x19
            p3 = self._upsample_add(p4, self.latlayer2(p3)) # 38x38
            sources = [p3, p4, p5, p6, p7]
            # 2x BiFPNs for D0
            sources = self.BiFPN1(sources)
            sources = self.BiFPN2(sources)
            # wrap outputs.
            sources = [sources[0], sources[1], sources[2], sources[3], sources[4], p8]
        
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
    
