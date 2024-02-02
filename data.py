import os
import random
import json
import boto3
import torch
import sagemaker

from collections import defaultdict
from typing import List, Tuple, Any, Optional, Callable
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class CocoDataset(Dataset):
    
    def __init__(
        self,
        image_path: str,
        text_path: str,
        split: str,
        image_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        
        ann_path = f'targets_{split}.json'
        dataset = json.load(open(os.path.join(text_path, ann_path), 'r', encoding='utf-8'))
        
        anns = {}
        imgToAnns = defaultdict(list)
        for ann in dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann['id'])
            anns[ann['id']] = ann
            
        self.anns = anns
        self.imgToAnns = imgToAnns
            
        self.imgs = {img['id']:img for img in dataset['images']}
        self.ids = list(sorted(self.imgs.keys()))
        
        self.image_path = image_path
        
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.split = split

    def _load_image(self, id:int) -> Image.Image:
        path = os.path.join(self.image_path, self.imgs[id]['file_name'])
        return Image.open(path).convert('RGB')
    
    def _load_target(self, id:int) -> List[str]:
        targets = []
        for ann_id in self.imgToAnns[id]:
            target = self.anns[ann_id]['target']
            if target:
                targets.append(target)
                
        return random.choice(targets)

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        
        PIL_to_Tensor = ToTensor()
        image = PIL_to_Tensor(image)
        
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return (image, target)
        
    def __len__(self) -> int:
        return len(self.ids)
    
def collate_fn(batch):
    images, captions = zip(*batch)
    images, captions = torch.stack(images, 0), torch.tensor(captions)
    return (images, captions)
    
        