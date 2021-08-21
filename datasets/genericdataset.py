import os
from PIL import Image

import torch
import torch.utils.data as data

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def imresize(img, imsize):
    img.thumbnail((imsize, imsize), Image.ANTIALIAS)
    return img


class ImagesFromList(data.Dataset):
    def __init__(self, root, images, imsize=None, bbxs=None, transform=None, loader=default_loader):
        images_fn = [os.path.join(root,images[i]) for i in range(len(images))]
        if len(images_fn) == 0:
            raise(RuntimeError("Dataset contains 0 images!"))
        self.root = root
        self.images = images
        self.imsize = imsize
        self.images_fn = images_fn
        self.bbxs = bbxs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.images_fn[index]
        img = self.loader(path)
        imfullsize = max(img.size)

        if self.bbxs is not None:
            img = img.crop(self.bbxs[index])
        if self.imsize is not None:
            if self.bbxs is not None:
                img = imresize(img, self.imsize * max(img.size) / imfullsize)
            else:
                img = imresize(img, self.imsize)
        if self.transform is not None:
            img = self.transform(img)

        return img, path

    def __len__(self):
        return len(self.images_fn)
