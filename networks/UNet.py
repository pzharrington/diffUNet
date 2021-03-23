import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicPad2d(nn.Module):

    def __init__(self, pad, periodic=True):
        super().__init__()
        self.p = pad
        self.mode = 'circular' if periodic else 'constant'

    def forward(self, x):
        return F.pad(x, pad=(self.p, self.p, self.p, self.p), mode=self.mode)


def DownBlock(params, in_channels, out_channels):
    """Downsampling block"""
    layers = []
    inC = in_channels
    outC = out_channels//2

    # stride 1 convs
    for i in range(params.N_stride1_convs):
        layers.append(PeriodicPad2d(pad=1, periodic=params.use_periodic_padding))
        layers.append(nn.Conv2d(inC, outC, 3, stride=1, padding=0))
        if params.useBN:
            layers.append(nn.BatchNorm2d(outC))
        layers.append(nn.LeakyReLU(negative_slope=params.LeakyReLU_alpha))
        inC = outC

    # downsampling stride 2 conv
    layers.append(PeriodicPad2d(pad=1, periodic=params.use_periodic_padding))
    layers.append(nn.Conv2d(inC, out_channels, params.kernel_size, stride=2, padding=0))
    if params.useBN:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(negative_slope=params.LeakyReLU_alpha))
    return nn.Sequential(*layers)


def UpBlock(params, cat_channels, in_channels, out_channels, isLast=False):
    """Usampling block"""
    layers = []
    inC = in_channels+cat_channels
    outC = in_channels

    # stride 1 convs
    for i in range(params.N_stride1_convs):
        layers.append(PeriodicPad2d(pad=1, periodic=params.use_periodic_padding))
        layers.append(nn.Conv2d(inC, outC, 3, stride=1, padding=0))
        if params.useBN:
            layers.append(nn.BatchNorm2d(outC))
        layers.append(nn.LeakyReLU(negative_slope=params.LeakyReLU_alpha))
        inC = outC

    # upsampling stride 2 conv
    layers.append(nn.ConvTranspose2d(inC, out_channels, params.kernel_size, stride=2, padding=1, output_padding=0))
    if not isLast:
      if params.useBN:
          layers.append(nn.BatchNorm2d(out_channels))
      layers.append(nn.LeakyReLU(negative_slope=params.LeakyReLU_alpha))
    elif params.final_act:
      act = {'tanh': nn.Tanh(), 'relu':nn.ReLU()}
      layers.append(act[params.final_act])
    return nn.Sequential(*layers)


def loss_func(gen_output, target, params):
    l1_loss = nn.functional.l1_loss(gen_output, target)
    return l1_loss



class UNet(nn.Module):

    def __init__(self, params):
        super().__init__()

        self.N_scales = params.N_feature_scales

        inC = params.N_channels
        outC = params.Nf_base
        
        # Build dowsampling path
        down = []
        for i in range(self.N_scales):
            down.append(DownBlock(params, inC, outC))
            inC = outC
            outC *= 2

        # Build upsampling path
        catC = 0
        outC //= 4
        up = []
        for i in range(self.N_scales - 1):
            up.append(UpBlock(params, catC, inC, outC))
            inC = outC
            catC = outC
            outC //= 2
        up.append(UpBlock(params, catC, inC, params.N_channels, isLast=True))
        
        # Add down/up-sampling paths to model
        self.down = nn.ModuleList(down)
        self.up = nn.ModuleList(up)
        

    def forward(self, x):
        
        skips = []
        for d in self.down:
            x = d(x)
            skips.append(x)
        skips = reversed(skips[:-1]) 
        for skip, u in zip(skips, self.up[:-1]):
            x = u(x)
            x = torch.cat([skip, x], dim=1)
        out = self.up[-1](x)
        return out


    def get_weights_function(self, params):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, params['conv_scale'])
                if params['conv_bias'] is not None:
                    m.bias.data.fill_(params['conv_bias'])
        return weights_init
 
