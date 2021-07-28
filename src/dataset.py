import random
import cv2
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

### copy from https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
class CutoutV2(A.DualTransform):
    def __init__(
        self,
        num_holes=8,
        max_h_size=8,
        max_w_size=8,
        fill_value=0,
        always_apply=False,
        p=0.5,
    ):
        super(CutoutV2, self).__init__(always_apply, p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value

    def apply(self, image, fill_value=0, holes=(), **params):
        return A.functional.cutout(image, holes, fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _n in range(self.num_holes):
            y = random.randint(0, height)
            x = random.randint(0, width)

            y1 = np.clip(y - self.max_h_size // 2, 0, height)
            y2 = np.clip(y1 + self.max_h_size, 0, height)
            x1 = np.clip(x - self.max_w_size // 2, 0, width)
            x2 = np.clip(x1 + self.max_w_size, 0, width)
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("num_holes", "max_h_size", "max_w_size")   
    
def get_train_transforms(CFG):
    return A.Compose([   
            ##copy from https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
            A.Resize(CFG.image_size, CFG.image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightness(limit=0.2, p=0.75),
            A.RandomContrast(limit=0.2, p=0.75),
            A.OneOf([
                A.OpticalDistortion(distort_limit=1.),
                A.GridDistortion(num_steps=5, distort_limit=1.),
            ], p=0.75),

            A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, border_mode=0, p=0.75),
            CutoutV2(max_h_size=int(CFG.image_size * 0.4), max_w_size=int(CFG.image_size * 0.4), num_holes=1, p=0.75),
        
        
#             A.RandomResizedCrop(height=CFG.image_size, width=CFG.image_size, scale=(0.85,1), p=1.0),
#             A.Resize(CFG.image_size, CFG.image_size),
#             # A.Transpose(p=0.5),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             # A.RandomRotate90(p=0.5),
#             A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
#             A.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.5),
#             A.ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.20,rotate_limit=20, p=0.5),
            # A.CenterCrop(CFG.image_size, CFG.image_size),
        
            ### heavy augs
            # A.CLAHE(clip_limit=(1,4),p=0.5),
            # A.OneOf([
            #     A.OpticalDistortion(distort_limit=1.0),
            #     A.GridDistortion(num_steps=5,distort_limit=1.0),
            #     A.ElasticTransform(alpha=3),
            # ],p=0.2),
            # A.OneOf([
            #     A.GaussNoise(var_limit=[10,50]),
            #     A.GaussianBlur(),
            #     A.MotionBlur(),
            #     A.MedianBlur(),
            # ],p=0.2),
            # A.Resize(CFG.image_size,CFG.image_size),
            # A.OneOf([
            #     A.JpegCompression(),
            #     A.Downscale(scale_min=0.1, scale_max=0.15),
            # ],p=0.2),
            # A.IAAPiecewiseAffine(p=0.2),
            # A.IAASharpen(p=0.2),
#             A.CoarseDropout(p=0.5),
#             A.Cutout(max_h_size=16, max_w_size=16, fill_value=(0.,0.,0.), num_holes=16, p=0.5),
        
            ###for cls without mask
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
#             ToTensorV2(),
        ],p=1.0)

def get_val_transforms(CFG):
    return A.Compose([
            A.Resize(CFG.image_size, CFG.image_size),
            ###for cls without mask
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
#             ToTensorV2(),
        ],p=1.0)

def to_tensor(x, **kwargs):
    if x.ndim==2 : 
        x = np.expand_dims(x,2)
    x = np.transpose(x,(2,0,1)).astype('float32') / 255.
    
    x = torch.from_numpy(x)
    return x

def get_preprocessing():
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
#         A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)

class SIIMDataset(Dataset):
    def __init__(self, CFG, df, transforms=None, output_label=True):
        super().__init__()
        self.CFG = CFG
        self.df = df
        self.transforms = transforms
        self.output_label = output_label
        self.length = len(df)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        image_path = self.CFG.train_dir + '%s.png' % (d.image)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            raise FileNotFoundError(image_path)
            
        ### do cutout
#         if self.do_cutout and random.random() < self.CFG.cutout_prob:
#             image = cutout(image)
        
        label = d.label
        
        # apply augmentations
        if self.transforms:
            image = self.transforms(image=image)['image']
        else:
            image = torch.from_numpy(image)
        
        if self.output_label:
            return image, label
        else:
            return image
        
class SIIMMaskDataset(Dataset):
    def __init__(self, CFG, df, do_cutout=False, transforms=None, preprocessing=None, output_label=True):
        super().__init__()
        self.CFG = CFG
        self.df = df
        self.do_cutout = do_cutout
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.output_label = output_label
        self.length = len(df)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        image_path = self.CFG.train_dir + '%s.png' % (d.image)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_path = self.CFG.mask_dir + '%s.png' % (d.image)
        mask = cv2.imread(mask_path, 0)
        
        if image is None:
            raise FileNotFoundError(image_path)
            
        ### do cutout
        if self.do_cutout and random.random() < self.CFG.cutout_prob:
            image = cutout(image)
        
        label = d.label  
        # apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented['image'],augmented['mask']
            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.output_label:
            return image, mask, label
        else:
            return image, mask
        
class SIIMMaskExtDataset(Dataset):
    def __init__(self, CFG, df, do_cutout=False, transforms=None, preprocessing=None, output_label=True):
        super().__init__()
        self.CFG = CFG
        self.df = df
        self.do_cutout = do_cutout
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.output_label = output_label
        self.length = len(df)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        image_path = self.CFG.external_train_dir + '%s' % (d.fname.replace('.dcm',''))
#         print(f'external image_path is {image_path}')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         mask_path = self.CFG.external_mask_dir + '%s' % (d.id.replace('.dcm.jpg','.png'))
#         print(f'external mask_path is {mask_path}')
        mask = np.zeros((self.CFG.image_size, self.CFG.image_size), dtype=np.uint8)
        
        if image is None:
            raise FileNotFoundError(image_path)
            
        ### do cutout
        if self.do_cutout and random.random() < self.CFG.cutout_prob:
            image = cutout(image)
        
        label = d.pred_label  
        # apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented['image'],augmented['mask']
            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.output_label:
            return image, mask, label
        else:
            return image, mask
        
class SIIMMaskPseudoDataset(Dataset):
    def __init__(self, CFG, df, do_cutout=False, transforms=None, preprocessing=None, output_label=True):
        super().__init__()
        self.CFG = CFG
        self.df = df
        self.do_cutout = do_cutout
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.output_label = output_label
        self.length = len(df)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        split = d.id.split('_')[-1]
        if split == 'study':
            image_path = self.CFG.pseudo_train_dir + '/study/%s.png' % (d.id)
        else:
            image_path = self.CFG.pseudo_train_dir+ '/image/%s.png' % (d.id)
#         image_path = self.CFG.pseudo_train_dir + '%s' % (d.id.replace('.dcm',''))
#         print(f'external image_path is {image_path}')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         mask_path = self.CFG.external_mask_dir + '%s' % (d.id.replace('.dcm.jpg','.png'))
#         print(f'external mask_path is {mask_path}')
        mask = np.zeros((self.CFG.image_size, self.CFG.image_size), dtype=np.uint8)
        
        if image is None:
            raise FileNotFoundError(image_path)
            
        ### do cutout
        if self.do_cutout and random.random() < self.CFG.cutout_prob:
            image = cutout(image)
        
        label = d.label  
        # apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented['image'],augmented['mask']
            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.output_label:
            return image, mask, label
        else:
            return image, mask        

class SIIM6CLSMaskDataset(Dataset):
    def __init__(self, CFG, df, do_cutout=False, transforms=None, preprocessing=None, output_label=True):
        super().__init__()
        self.CFG = CFG
        self.df = df
        self.do_cutout = do_cutout
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.output_label = output_label
        self.length = len(df)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        image_path = self.CFG.train_dir + '%s.png' % (d.image)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_path = self.CFG.mask_dir + '%s.png' % (d.image)
        mask = cv2.imread(mask_path, 0)
        
        if image is None:
            raise FileNotFoundError(image_path)
            
        ### do cutout
        if self.do_cutout and random.random() < self.CFG.cutout_prob:
            image = cutout(image)
        
#         label = d.label  
        onehot = torch.from_numpy(np.array(d[self.CFG.study_name].values).astype(np.float32))
#         print(f'onehot is {onehot}')
        # apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented['image'],augmented['mask']
            
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.output_label:
            return image, mask, onehot
        else:
            return image, mask
        
def siim_collate(batch):
    imgs, masks, targets = [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        masks.append(data_point[1])
        targets.append(data_point[2])

    return torch.stack(imgs), torch.stack(masks), torch.stack(targets)
