import torch
import torch.nn as nn

class HLKConv(nn.Module):
    def __init__(self, dim, k_size):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=((3-1)//2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=((3-1)//2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=4, groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=((5-1)//2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=((5-1)//2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=11, stride=1, padding=15, groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=((5-1)//2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=13, stride=1, padding=18, groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=((5-1)//2), groups=dim)
            self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=17, stride=1, padding=24, groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim * 2, dim, 1)


    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(torch.cat([x, self.conv_spatial(x)], dim=1))
        return x
    
if __name__ == '__main__':
    x = torch.randn((2, 2, 512, 512))
    model = HLKConv(2, 53)
    print(model(x).shape)