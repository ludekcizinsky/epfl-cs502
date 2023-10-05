import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

'''
Nothing needs to be changed in this file unless you want to play around.
'''

def get_indices(length, dataset_used, data_split, new=False):
    """ 
    Gets the Training & Testing data indices
    """

    # Pickle file location of the indices.
    file_path = os.path.join('dataset',f'split_indices_{dataset_used}.p')
    data = dict()
    
    if os.path.isfile(file_path) and not new:
        # File found.
        with open(file_path,'rb') as file :
            data = pickle.load(file)
            return data['train_indices'], data['test_indices']
        
    else:
        # File not found or fresh copy is required.
        indices = list(range(length))
        np.random.shuffle(indices)
        split = int(np.floor(data_split * length))
        train_indices , test_indices = indices[split:], indices[:split]

        # Indices are saved with pickle.
        data['train_indices'] = train_indices
        data['test_indices'] = test_indices
        with open(file_path,'wb') as file:
            pickle.dump(data,file)
    return train_indices, test_indices

def result(image, mask, output, title, transparency=0.38, save_path=None):
    """ 
    Plots a 2x3 plot with comparisons of output and original image.
    """

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(
        20, 15), gridspec_kw={'wspace': 0.025, 'hspace': 0.010})
    fig.suptitle(title, x=0.5, y=0.92, fontsize=20)

    axs[0][0].set_title("Original Mask", fontdict={'fontsize': 16})
    axs[0][0].imshow(mask, cmap='gray')
    axs[0][0].set_axis_off()

    axs[0][1].set_title("Constructed Mask", fontdict={'fontsize': 16})
    axs[0][1].imshow(output, cmap='gray')
    axs[0][1].set_axis_off()

    mask_diff = np.abs(np.subtract(mask, output))
    axs[0][2].set_title("Mask Difference", fontdict={'fontsize': 16})
    axs[0][2].imshow(mask_diff, cmap='gray')
    axs[0][2].set_axis_off()

    seg_output = mask*transparency
    seg_image = np.add(image, seg_output)/2
    axs[1][0].set_title("Original Segment", fontdict={'fontsize': 16})
    axs[1][0].imshow(seg_image, cmap='gray')
    axs[1][0].set_axis_off()

    seg_output = output*transparency
    seg_image = np.add(image, seg_output)/2
    axs[1][1].set_title("Constructed Segment", fontdict={'fontsize': 16})
    axs[1][1].imshow(seg_image, cmap='gray')
    axs[1][1].set_axis_off()

    axs[1][2].set_title("Original Image", fontdict={'fontsize': 16})
    axs[1][2].imshow(image, cmap='gray')
    axs[1][2].set_axis_off()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=90, bbox_inches='tight')

    plt.show()

class DiceLoss(nn.Module):
    """
    Sørensen–Dice coefficient loss.

    To know more about this loss check this link:
    https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predicted, target):
        
        batch = predicted.size()[0]
        batch_loss = 0
        for index in range(batch):
            coefficient = self._dice_coefficient(
                predicted[index], target[index])
            batch_loss += coefficient

        batch_loss = batch_loss / batch

        return 1 - batch_loss

    def _dice_coefficient(self, predicted, target):
        
        smooth = 1
        product = torch.mul(predicted, target)
        intersection = product.sum()
        coefficient = (2*intersection + smooth) / (predicted.sum() + target.sum() + smooth)
        return coefficient

class BCEDiceLoss(nn.Module):
    """ 
    Combination of Binary Cross Entropy Loss and Soft Dice Loss.
    """

    def __init__(self, device):
        
        super(BCEDiceLoss, self).__init__()
        self.dice_loss = DiceLoss().to(device)

    def forward(self, predicted, target):
        
        return F.binary_cross_entropy(predicted, target) + self.dice_loss(predicted, target)
