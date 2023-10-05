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
        self.conv1_1 = ...
        self.conv1_2 = ...
        self.maxpool1 = ...

        # Block 2
        self.conv2_1 = ...
        self.conv2_2 = ...
        self.maxpool2 = ...

        # Block 3
        self.conv3_1 = ...
        self.conv3_2 = ...
        self.maxpool3 = ...

        # Block 4
        self.conv4_1 = ...
        self.conv4_2 = ...
        self.maxpool4 = ...
        
        # ToDo 2: Bottleneck Part of Network.
        # Hint: 
        # 1. You only need two convolution layers.
        self.conv5_1 = ...
        self.conv5_2 = ...

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
        self.conv6_up = ...
        self.conv6_1 = ...
        self.conv6_2 = ...
        
        # Block 3
        self.conv7_up = ...
        self.conv7_1 = ...
        self.conv7_2 = ...
        
        # Block 2
        self.conv8_up = ...
        self.conv8_1 = ...
        self.conv8_2 = ...
        
        # Block 1
        self.conv9_up = ...
        self.conv9_1 = ...
        self.conv9_2 = ...

        # ToDo 4: Output Part of Network.
        self.conv10 = ...

    def forward(self, x):
        """ 
        Forward propagation of the network.
        """

        # ToDo 5: Encoding Part of Network.
        # Hint: Do not forget to add activate function, e.g. ReLU, between to convolution layers. (same for the bottlenect and decoder)

        #   Block 1
        
        #   Block 2
        
        #   Block 3
        
        #   Block 4

        # ToDo 6: Bottleneck Part of Network.

        # ToDo 7: Decoding Part of Network.
        # Hint: 
        # 1. Do not forget to concatnate the outputs from the previous decoder and the corresponding encoder.
        # 2. You can try torch.cat() to concatnate the tensors.

        #   Block 4

        #   Block 3

        #   Block 2

        #   Block 1

        # ToDo 8: Output Part of Network.
        # Hint: 
        # 1. Our task is a binary segmentation, and it is to classify each pixel into foreground or backfround. 
        # 2. Sigmoid is a useful activate function for binary classification.
        
        output = ...

        return output
