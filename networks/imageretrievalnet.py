import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from datasets.genericdataset import ImagesFromList

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class L2N(nn.Module):
    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

class ImageRetrievalNet(nn.Module):
    def __init__(self, features, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta

    def forward(self, x):
        o = self.features(x)
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)
        if self.whiten is not None:
            o = self.norm(self.whiten(o))
        return o.permute(1, 0)

def init_network(params):
    architecture = params.get('architecture', 'resnet50')
    pooling = params.get('pooling', 'gem')
    whitening = params.get('whitening', False)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    dim = 2048
    net_in = getattr(torchvision.models, architecture)(pretrained=False)
    features = list(net_in.children())[:-2]
    pool = GeM()
    whiten = nn.Linear(dim, dim, bias=True)
    meta = {
        'architecture': architecture,
        'pooling': pooling,
        'whitening': whitening,
        'mean': mean,
        'std': std,
        'outputdim': dim,
    }
    net = ImageRetrievalNet(features, pool, whiten, meta)
    return net

def extract_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1):
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    with torch.no_grad():
        vecs = torch.zeros(net.meta['outputdim'], len(images))
        img_paths = list()
        for i, (input, path) in enumerate(loader):
            if torch.cuda.is_available():
                input = input.cuda()
            if len(ms) == 1 and ms[0] == 1:
                vecs[:, i] = extract_ss(net, input)
            else:
                vecs[:, i] = extract_ms(net, input, ms, msp)
            img_paths.append(path)

    return vecs, img_paths


def extract_ss(net, input):
    return net(input).cpu().data.squeeze()


def extract_ms(net, input, ms, msp):
    v = torch.zeros(net.meta['outputdim'])

    for s in ms:
        if s == 1:
            input_t = input.clone()
        else:
            input_t = nn.functional.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
        v += net(input_t).pow(msp).cpu().data.squeeze()

    v /= len(ms)
    v = v.pow(1. / msp)
    v /= v.norm()

    return v
