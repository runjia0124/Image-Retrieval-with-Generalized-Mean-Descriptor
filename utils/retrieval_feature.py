import os
from PIL import Image
from lshash.lshash import LSHash
import torch
from torchvision import transforms
from networks.imageretrievalnet import init_network, extract_vectors
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

class ImageProcess():
    def __init__(self, img_dir):
        self.img_dir = img_dir

    def process(self):
        imgs = list()
        for root, dirs, files in os.walk(self.img_dir):
            for file in files:
                img_path = os.path.join(root + os.sep, file)
                try:
                    image = Image.open(img_path)
                    if max(image.size) / min(image.size) < 5:
                        imgs.append(img_path)
                    else:
                        continue
                except:
                    print("image height/width ratio is small")
        return imgs

class AntiFraudFeatureDataset():
    def __init__(self, img_dir, network, feature_path='./feature', index_path='./index'):
        self.img_dir = img_dir
        self.network = network
        self.feature_path = feature_path
        self.index_path = index_path

    def constructfeature(self, hash_size, input_dim, num_hashtables):
        multiscale = '[1]'
        state = torch.load(self.network)
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['whitening'] = state['meta'].get('whitening', False)
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']

        net = init_network(net_params)
        net.load_state_dict(state['state_dict'])

        ms = list(eval(multiscale))

        if torch.cuda.is_available():
            net.cuda()
        net.eval()

        normalize = transforms.Normalize(
            mean=net.meta['mean'],
            std=net.meta['std']
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        images = ImageProcess(self.img_dir).process()
        vecs, img_paths = extract_vectors(net, images, 1024, transform, ms=ms)
        feature_dict = dict(zip(img_paths, list(vecs.detach().cpu().numpy().T)))

        lsh = LSHash(hash_size=int(hash_size), input_dim=int(input_dim), num_hashtables=int(num_hashtables))
        for img_path, vec in feature_dict.items():
            lsh.index(vec.flatten(), extra_data=img_path)

        with open(self.feature_path, "wb") as f:
            pickle.dump(feature_dict, f)
        with open(self.index_path, "wb") as f:
            pickle.dump(lsh, f)

        return feature_dict, lsh

    def test_feature(self):
        multiscale = '[1]'
        state = torch.load(self.network)
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['whitening'] = state['meta'].get('whitening', False)
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']

        net = init_network(net_params)
        net.load_state_dict(state['state_dict'])

        ms = list(eval(multiscale))

        if torch.cuda.is_available():
            net.cuda()
        net.eval()

        normalize = transforms.Normalize(
            mean=net.meta['mean'],
            std=net.meta['std']
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        images = ImageProcess(self.img_dir).process()
        vecs, img_paths = extract_vectors(net, images, 1024, transform, ms=ms)
        feature_dict = dict(zip(img_paths, list(vecs.detach().cpu().numpy().T)))

        return feature_dict
