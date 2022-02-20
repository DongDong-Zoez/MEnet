class ConvLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, num_conv):
        super(ConvLayer, self).__init__()
        
        def convReLU(self, in_channels, out_channels, kernel_size):
            return nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                        #nn.BatchNorm2d(out_channels, eps=1e-5),
                        nn.ReLU(inplace=True),
                  )
        
        self.conv =  nn.Sequential()
        for i in range(num_conv):
            if i == 0:
                self.conv.add_module('conv_%d'%i, convReLU(self, in_channels, out_channels, kernel_size))
            else:
                self.conv.add_module('conv_%d'%i, convReLU(self, out_channels, out_channels, kernel_size))
                
        self.conv.add_module('MaxPooling', nn.MaxPool2d(2))
        
    def forward(self, x):
        return self.conv(x)
    
class FCN(nn.Module):
    
    def __init__(self, mode='origin'):
        super(FCN, self).__init__()

        self.mode = mode
        in_channels=3
        output_channel=3
        out_channels=[64, 128, 256, 512, 512]
        if self.mode == 'Large':
            out_channels=[128, 256, 512, 768, 768]
        num_conv = [2, 2, 3, 3, 3]
        kernel_size=3
        scale = 8
        
        self.l1 = ConvLayer(in_channels, out_channels[0], kernel_size, num_conv[0])
        self.l2 = ConvLayer(out_channels[0], out_channels[1], kernel_size, num_conv[1])
        self.l3 = ConvLayer(out_channels[1], out_channels[2], kernel_size, num_conv[2])
        self.l4 = ConvLayer(out_channels[2], out_channels[3], kernel_size, num_conv[3])
        self.l5 = ConvLayer(out_channels[3], out_channels[4], kernel_size, num_conv[4])

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.output = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

        self.sigmoid = nn.ReLU()
        self.conv4 = nn.Conv2d(out_channels[3], out_channels[2], kernel_size=1)
        self.conv3 = nn.Conv2d(out_channels[2], out_channels[2], kernel_size=1)
        self.conv = nn.Conv2d(out_channels[2], output_channel, kernel_size=1)
        
    def forward(self, x):
        
        x = self.l1(x)
        x = self.l2(x)
        x3 = self.l3(x)
        x4 = self.l4(x3)
        x5 = self.l5(x4)

        x4 = x4 + self.up(x5)
        x4 = self.sigmoid(self.conv4(x4))
        x3 = x3 + self.up(x4)
        x3 = self.sigmoid(self.conv3(x3))

        x = self.conv(self.output(x3))

        return x
