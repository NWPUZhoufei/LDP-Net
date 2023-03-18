import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import ImageFile
from PIL import ImageEnhance

ImageFile.LOAD_TRUNCATED_IMAGES = True
identity = lambda x:x
transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]
        
    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')
        return out
    

    
    
class SetDataset:
    def __init__(self, data_path, num_class, batch_size, transform):
        self.sub_meta = {}
        self.data_path = data_path
        self.num_class = num_class
        self.cl_list = range(self.num_class)
        for cl in self.cl_list:
            self.sub_meta[cl] = []
        d = ImageFolder(self.data_path)
        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)

    
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):

        img = self.transform(self.sub_meta[i])
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]
    

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param = dict(mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']
        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class Eposide_DataManager():
    def __init__(self, data_path, num_class, image_size, n_way=5, n_support=1, n_query=15, n_eposide=1):        
        super(Eposide_DataManager, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.data_path, self.num_class, self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)  
        data_loader_params = dict(batch_sampler=sampler, num_workers=12, pin_memory=True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader




        

        
