import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """ 
    For more information about U-Net Architecture check the paper here.
    Link :- https://arxiv.org/abs/1505.04597
    """

    def __init__(self, filter_num, input_channels=1, output_channels=1):

        """ Constructor for UNet class.
        Parameters:
            filter_num: A list of number of filters (number of input or output channels of each layer).
            input_channels: Input channels for the network.
            output_channels: Output channels for the final network.
        """

        # Since UNet is a child class, we need to first call the __init__ function of its parent class
        super(UNet, self).__init__()

        # We set hyper-parameter padding and kernel size
        padding = 1
        ks = 3

        # ToDo 1: Encoding Part of Network.
        # Hint: 
        # 1. For each block, you need two convolution layers and one maxpooling layer.
        # 2. input_channels is the number of the input channels. The filter_num is a list of number of filters. 
        #    For example, in block 1, the input channels should be input_channels, and the output channels should be filter_num[0];
        #    in block 2, the input channels should be filter_num[0], and the output channels should be filter_num[1]
        # 3. torch.nn contains the layers you need. You can check torch.nn from link: https://pytorch.org/docs/stable/nn.html

        # Block 1
        self.conv1_1 = nn.Conv2d(input_channels, filter_num[0], kernel_size=ks, stride=1, padding=padding)
        self.conv1_2 = nn.Conv2d(filter_num[0], filter_num[0], kernel_size=ks, stride=1, padding=padding)
        self.maxpool1 = nn.MaxPool2d(kernel_size=ks - 1)
 
        # Block 2
        self.conv2_1 = nn.Conv2d(filter_num[0], filter_num[1], kernel_size=ks, stride=1, padding=padding)
        self.conv2_2 = nn.Conv2d(filter_num[1], filter_num[1], kernel_size=ks, stride=1, padding=padding)
        self.maxpool2 = nn.MaxPool2d(kernel_size=ks - 1)

        # Block 3
        self.conv3_1 = nn.Conv2d(filter_num[1], filter_num[2], kernel_size=ks, stride=1, padding=padding)
        self.conv3_2 = nn.Conv2d(filter_num[2], filter_num[2], kernel_size=ks, stride=1, padding=padding)
        self.maxpool3 = nn.MaxPool2d(kernel_size=ks - 1)

        # Block 4
        self.conv4_1 = nn.Conv2d(filter_num[2], filter_num[3], kernel_size=ks, stride=1, padding=padding)
        self.conv4_2 = nn.Conv2d(filter_num[3], filter_num[3], kernel_size=ks, stride=1, padding=padding)
        self.maxpool4 = nn.MaxPool2d(kernel_size=ks - 1)
        
        # ToDo 2: Bottleneck Part of Network.
        # Hint: 
        # 1. You only need two convolution layers.
        self.conv5_1 = nn.Conv2d(filter_num[3], filter_num[4], kernel_size=ks, stride=1, padding=padding)
        self.conv5_2 = nn.Conv2d(filter_num[4], filter_num[4], kernel_size=ks, stride=1, padding=padding)

        # ToDo 3: Decoding Part of Network.
        # Hint: 
        # 1. For each block, you need one upsample+convolution layer and two convolution layers.
        # 2. output_channels is the number of the output channels. The filter_num is a list of number of filters. 
        #    However, we need to use it reversely.
        #    For example, in block 4 of decoder, the input channels should be filter_num[4], and the output channels should be filter_num[3];
        #    in Output Part of Network, the input channels should be filter_num[0], and the output channels should be output_channels.
        # 3. torch.nn contains the layers you need. You can check torch.nn from link: https://pytorch.org/docs/stable/nn.html
        # 4. Using nn.ConvTranspose2d is one way to do upsampling and convolution at the same time.

        # Block 4
        self.conv6_up = nn.ConvTranspose2d(filter_num[4], filter_num[3], kernel_size=ks - 1, stride=2)
        self.conv6_1 = nn.Conv2d(filter_num[4], filter_num[3], kernel_size=ks, stride=1, padding=padding)
        self.conv6_2 = nn.Conv2d(filter_num[3], filter_num[3], kernel_size=ks, stride=1, padding=padding)
        
        # Block 3
        self.conv7_up = nn.ConvTranspose2d(filter_num[3], filter_num[2], kernel_size=ks - 1, stride=2)
        self.conv7_1 = nn.Conv2d(filter_num[3], filter_num[2], kernel_size=ks, stride=1, padding=padding)
        self.conv7_2 = nn.Conv2d(filter_num[2], filter_num[2], kernel_size=ks, stride=1, padding=padding)
        
        # Block 2
        self.conv8_up = nn.ConvTranspose2d(filter_num[2], filter_num[1], kernel_size=ks - 1, stride=2)
        self.conv8_1 = nn.Conv2d(filter_num[2], filter_num[1], kernel_size=ks, stride=1, padding=padding)
        self.conv8_2 = nn.Conv2d(filter_num[1], filter_num[1], kernel_size=ks, stride=1, padding=padding)
        
        # Block 1
        self.conv9_up = nn.ConvTranspose2d(filter_num[1], filter_num[0], kernel_size=ks - 1, stride=2)
        self.conv9_1 = nn.Conv2d(filter_num[1], filter_num[0], kernel_size=ks, stride=1, padding=padding)
        self.conv9_2 = nn.Conv2d(filter_num[0], filter_num[0], kernel_size=ks, stride=1, padding=padding)

        # ToDo 4: Output Part of Network.
        self.conv10 = nn.Conv2d(filter_num[0], output_channels, kernel_size=ks, stride=1, padding=padding)

    def forward(self, x):
        """ 
        Forward propagation of the network.
        """

        # ToDo 5: Encoding Part of Network.
        # Hint: Do not forget to add activate function, e.g. ReLU, between to convolution layers. (same for the bottlenect and decoder)

        #   Block 1
        x = F.relu(self.conv1_1(x))
        x = x1out = F.relu(self.conv1_2(x))
        x = self.maxpool1(x)
        # print(f"x1out: {x1out.shape}")

        #   Block 2
        x = F.relu(self.conv2_1(x))
        x = x2out = F.relu(self.conv2_2(x))
        x = self.maxpool2(x)
        # print(f"x2out: {x2out.shape}")
        #   Block 3
        x = F.relu(self.conv3_1(x))
        x = x3out = F.relu(self.conv3_2(x))
        x = self.maxpool3(x)       
        # print(f"x3out: {x3out.shape}")
        #   Block 4
        x = F.relu(self.conv4_1(x))
        x = x4out = F.relu(self.conv4_2(x))
        x = self.maxpool4(x)       
        # print(f"x4out: {x4out.shape}")
        # ToDo 6: Bottleneck Part of Network.
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        # print(f"bottleneck: {x.shape}")
        # ToDo 7: Decoding Part of Network.
        # Hint: 
        # 1. Do not forget to concatnate the outputs from the previous decoder and the corresponding encoder.
        # 2. You can try torch.cat() to concatnate the tensors.

        #   Block 4
        xup = self.conv6_up(x)
        # print(xup.shape, x4out.shape,)
        x = F.relu(self.conv6_1(torch.cat([x4out, xup], 1)))
        x = F.relu(self.conv6_2(x))

        #   Block 3
        xup = self.conv7_up(x)
        # print(xup.shape, x3out.shape,)
        x = F.relu(self.conv7_1(torch.cat([x3out, xup], 1)))
        x = F.relu(self.conv7_2(x))

        #   Block 2
        xup = self.conv8_up(x)
        # print(xup.shape, x2out.shape,)
        x = F.relu(self.conv8_1(torch.cat([x2out, xup], 1)))
        x = F.relu(self.conv8_2(x))

        #   Block 1
        xup = self.conv9_up(x)
        # print(xup.shape, x1out.shape,)
        x = F.relu(self.conv9_1(torch.cat([x1out, xup], 1)))
        x = F.relu(self.conv9_2(x))

        # ToDo 8: Output Part of Network.
        # Hint: 
        # 1. Our task is a binary segmentation, and it is to classify each pixel into foreground or backfround. 
        # 2. Sigmoid is a useful activate function for binary classification.
 
        output = torch.sigmoid(self.conv10(x))
        return output
