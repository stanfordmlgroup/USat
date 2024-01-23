import glob
import os
import random
import sys

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from usat.utils.builder import DATASET
from usat.utils.constants import IN_RGB_MEAN, IN_RGB_STD

LOC_SYNSET_MAP_FILE = 'LOC_synset_mapping.txt'
DATA_PATH = 'ILSVRC/Data/CLS-LOC/'

@DATASET.register_module()
class ImageNetDataset(Dataset):

    def __init__(self,
                 base_path,
                 split='train',
                 standardize=True,
                 load_type='all',
                 input_size=224,
                 **kwargs):
        self.base_path = base_path
        self.split = split
        self.standardize = standardize
        self.input_size = input_size
        self.synset_map = self.load_synset_to_dict(base_path)
        self.paths = self.load_file_paths(base_path, split, load_type, **kwargs)
        self.preprocessing = self.build_transform(standardize, input_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = torch.tensor(self.synset_map[path.split('/')[0]][0])
        img_fp = os.path.join(self.base_path, DATA_PATH, self.split, path)
        img = self.preprocessing(Image.open(img_fp).convert('RGB'))
        return img, label

    def load_file_paths(self, base_path, split, load_type, **kwargs):
        paths = []
        img_dir_fp = os.path.join(base_path, DATA_PATH, split)
        orig_dir = os.getcwd()
        all_img_classes = os.listdir(img_dir_fp)

        if load_type == 'all':
            os.chdir(img_dir_fp)
            glob_pattern = '*/*.JPEG'
            paths = glob.glob(glob_pattern)
            os.chdir(orig_dir)
            
        elif load_type == 'class':
            os.chdir(img_dir_fp)
            allowed_class_list = kwargs['class_list']
            for cl in allowed_class_list:
                paths.extend(glob.glob(f'{cl}/*.JPEG'))
            os.chdir(orig_dir)

        elif load_type == 'num_samples':
            num_samples = kwargs.get('num_samples')
            for cl in all_img_classes:
                all_imgs_for_cl = os.listdir(os.path.join(img_dir_fp, cl))
                # At most, sample the number of images in the class
                if len(all_imgs_for_cl) < num_samples:
                    num_samples = len(all_imgs_for_cl)
                imgs = random.sample(all_imgs_for_cl, num_samples)
                paths.extend([os.path.join(cl, img) for img in imgs])

        elif load_type == 'percent':
            percent = kwargs.get('percent')
            for cl in all_img_classes:
                all_imgs_for_cl = os.listdir(os.path.join(img_dir_fp, cl))
                # At most, sample the number of images in the class
                num_samples = int(len(all_imgs_for_cl) * percent)
                if num_samples > len(all_imgs_for_cl):
                    num_samples = len(all_imgs_for_cl)
                imgs = random.sample(all_imgs_for_cl, num_samples)
                paths.extend([os.path.join(cl, img) for img in imgs])

        else:
            print(f"Invalid load type for file paths: {load_type}")
            sys.exit(1)

        print(f"Found {len(paths)} data files")
        return paths

    def load_synset_to_dict(self, base_path):
        """ Return a mapping between synset and idx + label.
        {
            n01440764: (0, "tench, Tinca tinca"),
            n01443537: (1, "goldfish, Carassius auratus"),
            ...
        }
        """
        synset_map = {}
        synset_map_fp = os.path.join(base_path, LOC_SYNSET_MAP_FILE)
        with open(synset_map_fp, 'r') as f:
            for idx, line in enumerate(f):
                split_idx = line.find(' ')
                synset_map[line[0:split_idx]] = (idx, line[split_idx+1:])
        return synset_map

    def build_transform(self, standardize, input_size):
        t = []
        t.append(transforms.ToTensor())
        if standardize:
            t.append(transforms.Normalize(IN_RGB_MEAN, IN_RGB_STD))
        t.append(transforms.RandomResizedCrop(input_size, scale=(1.0, 1.0)))
        return transforms.Compose(t)
        

if __name__ == '__main__':
    random.seed(42)
    in_dataset = ImageNetDataset('/scr/imagenet/', split='train', load_type='all')
    #in_dataset = ImageNetDataset('/scr/imagenet/', split='train', load_type='class', class_list=['n12998815'])
    #in_dataset = ImageNetDataset('/scr/imagenet/', split='train', load_type='num_samples', num_samples=10)
    #in_dataset = ImageNetDataset('/scr/imagenet/', split='train', load_type='percent', percent=0.01)

