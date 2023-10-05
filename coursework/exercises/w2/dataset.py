from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

import os

'''
Here we try to build a Dataset class.
A common way is to define a Dataset class inherits from torch.utils.data.Dataset class.
Then, we override:
1. __init__: initialization
2. __getitem__: get a sample
3. __len__: get the length (number of samples) of the dataset
'''

class TumorDataset(Dataset):
    """ 
    A TumorDataset class.
    The object of this class represents tumor dataset.
    TumorDataset inherits from torch.utils.data.Dataset class.
    """

    def __init__(self, root_dir):

        """
        Inputs:
        1. root_dir: Directory with all the images.
        """

        # set root directory to root_dir
        self.root_dir = root_dir
        
        # The transform functions are to:
        # 1. Transform the data to the form that we need. E.g. use ToTensor to convert the datatype to Tensor
        # 2. Augment the data. E.g. flip and rotation.

        self.default_transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.RandomRotation(10),
            transforms.RandomVerticalFlip(0.5)
        ])

    def __getitem__(self, index):
        """
        __getitem__ is the function we need to override.
        Given an index, this function is to output the sample with that index.

        Here, a sample is a dictionary which contains:
        1. 'index': Index of the image.
        2. 'image': Contains the tumor image torch.Tensor.
        3. 'mask' : Contains the mask image torch.Tensor.
        """

        # load image and mask, give each of them an index, and return image, mask, and index.

        image_name = os.path.join(self.root_dir, str(index)+'.png')
        mask_name = os.path.join(self.root_dir, str(index)+'_mask.png')

        image = Image.open(image_name)
        mask = Image.open(mask_name)

        # Apply transform to both image and mask
        image = self.default_transformation(image)
        mask = self.default_transformation(mask)

        sample = {'index': int(index), 'image': image, 'mask': mask}

        return sample

    def __len__(self):
        """
        Returns the size of the dataset.
        """

        # ToDo 3: Get the size of the datasets (The number of samples in the dataset.)
        # Hint: The folder we provide contains samples and their mask, which means we have two images for each samples.
        
        file_count = len([f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))])
        size_of_dataset = int(file_count / 2)

        return size_of_dataset