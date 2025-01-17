import numpy as np
import torch
from torch import nn
import torchvision.transforms.v2 as transforms

np.random.seed(0)

""" GaussianBlur is Taken from https://github.com/sthalles/SimCLR/blob/master/data_aug/gaussian_blur.py """


class RandomChoiceExtended():
    def __init__(self, transforms_list, min_transforms, max_transforms):
        """
            FIXME ? IMPORTANT : DOESN'T GUARANTEE THAT DIFFERENT TRANSFORMATIONS ARE APPLIED EACH ITERATION
        """
        assert min_transforms >= 0 , "Min number of transforms must be bigger or equal to zero"
        assert max_transforms <= len(transforms_list), "Max number of transforms should be smaller than the number of available transforms"
        self.transform = transforms.RandomChoice(transforms_list)
        self.max_transforms = max_transforms
        self.min_transforms = min_transforms

    def __call__(self, img):
        for _ in range(np.random.randint(low=self.min_transforms, high=self.max_transforms + 1)):
            img = self.transform(img)
        return img

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        # self.pil_to_tensor = transforms.ToTensor()
        # self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        # img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(3, 6) # 0.1, 2.0
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        # img = self.tensor_to_pil(img)

        return img