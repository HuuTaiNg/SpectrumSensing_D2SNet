import torch.nn as nn
import torch
import torch.nn.functional as F


class SDM(nn.Module):
    def __init__(self):
        super(SDM, self).__init__()
        self.conv = nn.Conv2d(3, 256, kernel_size=1, padding=0, stride=1)
        self.PReLU = nn.PReLU()

        self.conv1_1 = nn.Conv2d(3, 128, kernel_size=5, padding=2, stride=1)
        self.bn1_1 = nn.BatchNorm2d(128)
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1)
        self.bn1_2 = nn.BatchNorm2d(128)
        self.conv1_3 = nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1)
        self.bn1_3 = nn.BatchNorm2d(128)
        self.conv1_4 = nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1)
        self.bn1_4 = nn.BatchNorm2d(128)
        self.conv1_5 = nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1)
        self.bn1_5 = nn.BatchNorm2d(128)

        self.conv2_1 = nn.Conv2d(3, 128, kernel_size=5, padding=2, stride=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.conv2_3 = nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1)
        self.bn2_3 = nn.BatchNorm2d(128)
        self.conv2_4 = nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1)
        self.bn2_4 = nn.BatchNorm2d(128)
        self.conv2_5 = nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1)
        self.bn2_5 = nn.BatchNorm2d(128)

        self.batchnorm = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(512, 128, kernel_size=5, padding=2, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 3, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x_org = x
        # Path 1
        x = self.PReLU(self.bn1_1(self.conv1_1(x)))
        x = self.PReLU(self.bn1_2(self.conv1_2(x)))
        left, right = torch.chunk(x, chunks=2, dim=3)
        x = torch.cat([right, left], dim=3)
        
        x = self.PReLU(self.bn1_3(self.conv1_3(x)))
        left, right = torch.chunk(x, chunks=2, dim=3)
        x = torch.cat([right, left], dim=3)  

        x = self.PReLU(self.bn1_4(self.conv1_4(x)))
        left, right = torch.chunk(x, chunks=2, dim=3)
        x = torch.cat([right, left], dim=3)  

        x = self.PReLU(self.bn1_5(self.conv1_5(x)))
        left, right = torch.chunk(x, chunks=2, dim=3)
        x = torch.cat([right, left], dim=3) 

        # Path 2
        x_r = torch.rot90(x_org, k=1, dims=[2, 3])
        x_r = self.PReLU(self.bn2_1(self.conv2_1(x_r)))
        x_r = self.PReLU(self.bn2_2(self.conv2_2(x_r)))
        left, right = torch.chunk(x_r, chunks=2, dim=3)
        x_r = torch.cat([right, left], dim=3)
        
        x_r = self.PReLU(self.bn2_3(self.conv2_3(x_r)))
        left, right = torch.chunk(x_r, chunks=2, dim=3)
        x_r = torch.cat([right, left], dim=3)  

        x_r = self.PReLU(self.bn2_4(self.conv2_4(x_r)))
        left, right = torch.chunk(x_r, chunks=2, dim=3)
        x_r = torch.cat([right, left], dim=3)  

        x_r = self.PReLU(self.bn2_5(self.conv2_5(x_r)))
        left, right = torch.chunk(x_r, chunks=2, dim=3)
        x_r = torch.cat([right, left], dim=3) 
        
        # ===================
        x_r = torch.rot90(x_r, k=-1, dims=[2, 3])
        x = torch.cat([x, x_r, self.PReLU(self.conv(x_org))], dim=1)

        x = self.batchnorm(x)
        x = self.PReLU(self.bn1(self.conv1(x)))
        x = self.PReLU(self.bn2(self.conv2(x)))
        x = F.sigmoid(self.conv3(x)) 

        return x
    

class SSM(nn.Module):
    def __init__(self, n_classes):
        super(SSM, self).__init__()     
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.prelu5 = nn.PReLU()
        self.prelu6 = nn.PReLU()
        self.prelu7 = nn.PReLU()

        self.bn = nn.BatchNorm2d(128)   
        self.conv1_1 = nn.Conv2d(3, 128, kernel_size=5, padding=2, stride=1)
        self.conv1_2 = nn.Conv2d(3, 128, kernel_size=5, padding=2, stride=1)

        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 6, kernel_size=5, padding=2, stride=1)
        self.bn7 = nn.BatchNorm2d(6)
        self.conv8 = nn.Conv2d(6, n_classes, kernel_size=5, padding=2, stride=1)

    def forward(self, x, x_freenoise):
        x = self.prelu1(self.conv1_1(x))
        x_freenoise = self.prelu1(self.conv1_2(x_freenoise))
        x = torch.cat([x, x_freenoise] ,dim=1)  
        x = self.bn1(x)
        x = self.prelu2(self.bn2(self.conv2(x))) 
        x = self.prelu3(self.bn3(self.conv3(x))) 
        x = self.prelu4(self.bn4(self.conv4(x))) 
        x = self.prelu5(self.bn5(self.conv5(x)))   
        x = self.prelu6(self.bn6(self.conv6(x))) 
        x = self.prelu7(self.bn7(self.conv7(x)))     
           
        return F.softmax(self.conv8(x), dim=1)