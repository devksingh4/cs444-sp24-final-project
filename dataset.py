from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import os
import albumentations as A
import numpy as np

_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)

class SegmentationDataset(Dataset):
    def __init__(self, root_dir="/data/contrails/", mode='train', should_transform=False):
        self.root_dir = root_dir
        self.mode = mode
        self.should_transform = should_transform
        self.id2label =  {'0': 'background', '1': 'contrails'}
        self.records = os.listdir(os.path.join(self.root_dir , self.mode))
        
#         if self.mode == 'train': 
#             select_cnt = 700
#             self.records = np.random.choice(self.records, select_cnt, replace=False)
        
#         if self.mode == 'validation':
#             select_cnt = 180
#             self.records = np.random.choice(self.records, select_cnt, replace=False)

    def __len__(self):
        return len(self.records)
    
    def transform(self, image, mask):
        aug = A.Compose([
                A.VerticalFlip(p=0.3),
                A.HorizontalFlip(p=0.3),      
                A.RandomRotate90(p=0.3),
                ]
            )

        random.seed(7)
        augmented = aug(image=image, mask=mask)

        image_transformed = augmented['image']
        mask_transformed = augmented['mask']
        
        return image_transformed, mask_transformed
    
    def normalize_range(self, data, bounds):
        return (data - bounds[0]) / (bounds[1] - bounds[0])

    def get_ash_img(self, bands):
        band11 = bands[:,:,0,11-8]
        band14 = bands[:,:,0,14-8]
        band15 = bands[:,:,0,15-8]
        r = self.normalize_range(band15 - band14, _TDIFF_BOUNDS)
        g = self.normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
        b = self.normalize_range(band14, _T11_BOUNDS)
        false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
        return false_color
        
    def __getitem__(self, idx):
        record_id = self.records[idx]
        record_dir = os.path.join(self.root_dir, self.mode, record_id)
        
        bands_data = []
        for i in range(8, 17):
            band_file = os.path.join(record_dir, f'band_{str(i).zfill(2)}.npy')
            band_data = np.load(band_file)
            bands_data.append(band_data)

        # Stack band data along the channel axis
        bands_data = np.stack(bands_data, axis=-1)
        ash = self.get_ash_img(bands_data)
        # If the data type is 'train' or 'validation', load the masks
        if self.mode in ['train', 'validation']:
            pixel_masks_file = os.path.join(record_dir, 'human_pixel_masks.npy')
            pixel_masks = np.load(pixel_masks_file)
            if self.should_transform:
                ash, pixel_masks = self.transform(ash, pixel_masks)
                
            sample = {'pixel_values': ash, 'labels': pixel_masks}
        else:
            sample = {'record_ids': str(record_dir.split('/')[-1]), 'pixel_values': ash}
      

        return sample
    
import torch
from glob import glob
import numpy as np
import os
class ContrailsDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.records = glob(f"{self.base_dir}/*/", recursive=False)
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        local_dir = os.path.join(self.base_dir, self.records[idx])
        band11 = np.load(f"{local_dir}band_11.npy")
        band14 = np.load(f"{local_dir}band_14.npy")
        band15 = np.load(f"{local_dir}band_15.npy")
        human_pixel_mask = np.load(f"{local_dir}human_pixel_masks.npy")
        x1 = torch.from_numpy(band11[:, :, 4])
        x2 = torch.from_numpy(band14[:, :, 4])
        x3 = torch.from_numpy(band15[:, :, 4])
        x = torch.stack((x1, x2, x3))
        y = torch.from_numpy(human_pixel_mask).permute(2, 0, 1).squeeze(dim=0).type(torch.LongTensor)
        return x, y