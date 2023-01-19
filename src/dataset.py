import torch
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
import glob

from PIL import Image
from plot import layer_plot, plot_grid

import random
import os


# TODO: Implement Random Rotate
# def rotate(p, origin=(0, 0), degrees=0):
#     angle = np.deg2rad(degrees)
#     R = np.array([[np.cos(angle), -np.sin(angle)],
#                   [np.sin(angle),  np.cos(angle)]])
#     o = np.atleast_2d(origin)
#     p = np.atleast_2d(p)
#     return np.squeeze((R @ (p.T-o.T) + o.T).T)



# class RandomRotate:
#     def __init__(self, angle):
#         self.angle = angle

#     def __call__(self, img, lab):
#         w, h = img.size
#         angle = random.randint(-self.angle, self.angle)
#         transforms.functional.rotate(img, angle, fill=img.mean())
#         points = np.zeros(9, w, 2)
#         for i in range(9):
#             for j in range(w):
#                 points[i,j,0] = j - w//2
#                 points[i,j,1] = lab[i,j] - h//2

#             points[i] = rotate(points[i], angle)

# TODO: RandomNoise to each pixel

class RandomHflip:

    def __call__(self, img, lab = None, mask = None):
        
        a = random.randint(0,1)
        if a == 1:
            img = transforms.functional.hflip(img)
            # if lab != None:
            new_lab = np.zeros(lab.shape)
            # print(lab.shape)
            for i in range(lab.shape[0]):
                for j in range(lab.shape[1]):
                    new_lab[i,j] = lab[i,lab.shape[1] -j -1 ]
            lab = new_lab

            if mask != None:
                mask = transforms.functional.hflip(mask)


        return (img, lab, mask)

class RandomNoise:
    def __init__(self, std, mean=0):
        self.mean = mean
        self.std = std

    def __call__(self, img, lab=None, mask=None):
        noise = torch.randn(img.shape)*self.std + self.mean
        img = img + noise
        return (img, lab, mask)

class Resize:
    def __init__(self, size=100):
        self.size = size

    def __call__(self, img, lab, mask):
        num_curves = lab.shape[0]
        fact = int(img.size[0]/self.size)
        # print(fact)

        img = transforms.functional.resize(img, self.size)
        # if lab != None:
        lab = lab/(float(fact))
        lab = lab.transpose(1,0)
        new_lab = np.zeros((self.size,num_curves))
        for i in range(self.size):
            new_lab[i] = lab[i*fact:(i+1)*fact].mean(axis=0)
        lab = new_lab.transpose(1,0)

        if mask != None:
            mask = transforms.functional.resize(mask, self.size)
        return (img, lab, mask)

class Normalize:
    def __init__(self, img_mean=0.0, img_std=1.0, lab_mean=0.0, lab_std=1.0):
        self.img_mean = img_mean
        self.img_std = img_std
        self.lab_mean = lab_mean
        self.lab_std = lab_std

    def __call__(self, img, lab, mask):
        img = (img - self.img_mean)/self.img_std
        lab = (lab - self.lab_mean)/self.lab_std
        return (img, lab, mask)

class RandomCrop:
    def __init__(self, ch, cw):
        self.ch = ch
        self.cw = cw

    def __call__(self, img, lab, mask):
        w, h = img.size
        low = 1
        high =0
        tries = 0
        while low > high:
            w_offset = random.randint(0, w-self.cw-1)
            new_lab = lab[:, w_offset: w_offset + self.cw]
            lab_max = lab.max()
            lab_min = lab.min()
            low = lab_max-self.ch
            high = min([h-self.ch, lab_min])
            # print(w,h, w_offset, lab_max, lab_min, high, low)
            tries += 1
            if tries > 100:
                print(w,h, w_offset, lab_max, lab_min, high, low)

            # print(tries)
        
        lab = new_lab
        h_offset = random.randint( low , high)
        lab = lab - h_offset

        img= transforms.functional.crop(img, h_offset, w_offset, self.ch, self.cw)

        if mask != None:
            mask= transforms.functional.crop(mask, h_offset, w_offset, self.ch, self.cw)

        return (img, lab, mask)

class RelOffset:

    def __call__(self, img, lab):
        nc, nw = lab.shape
        for i in range(1, nc):
            lab[i] = lab[i] - lab[i-1]
        return (img, lab)

class ToTensor:

    def __call__(self, img, lab, mask, lab_normalize=False):
        # if lab != None:
        lab = torch.FloatTensor(lab)
        if lab_normalize:
            lab = lab/float(img.size[0])
        
        return (transforms.functional.to_tensor(img), lab, torch.LongTensor(np.asarray(mask, dtype= np.uint8)))

class Compose:
    
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lab, mask=None):
        for t in self.transforms:
            img, lab, mask = t(img, lab, mask)
        return (img, lab, mask)

class EyeSegmentDataset(Dataset):
    def __init__(self, dataset_path="/home/g/gv53/Segmentation_AI/eye_seg_dataset", num_curves=2, transforms=transforms.ToTensor(), file_path=False, mask=False):
        self.transforms = transforms
        self.num_curves = num_curves
        self.img_paths = []
        self.mask = mask
        for i in range(1,11):
            for j in range(75):
                self.img_paths += glob.glob(dataset_path+'/' +str(i)+"/SKAN" + str(j) + '.bmp')
                self.img_paths += glob.glob(dataset_path+'/' +str(i)+"/skan" + str(j) + '.bmp')
    
        self.labels = np.zeros((10,75,9, 743))
        for i in range(10):
            loaded = np.load(dataset_path+'/'+str(i+1)+'/labels.npy')
            self.labels[i] = loaded

        self.ret_fp = file_path 

    def __getitem__(self, index):
        # print(index)
        img = Image.open(self.img_paths[index])
        path_components = self.img_paths[index].split('/')
        folder_index = int(path_components[-2])-1
        image_index_in_folder = int(path_components[-1].replace('.bmp','').replace('SKAN','').replace('skan', ''))
        label = self.labels[folder_index, image_index_in_folder,:self.num_curves]
        mask = None
        if self.mask:
            mask = Image.open(self.img_paths[index].replace('.bmp','_mask.png'))

        # print(img.size, mask.size, label.shape)
        if self.transforms != None:
            img, label, mask = self.transforms(img, label, mask) 
        label = label.transpose(1,0)
        # print(self.img_paths[index])
        if self.ret_fp:
            return (img, label, mask, self.img_paths[index])
        else:
            # print("size of img is", img.shape)
            # print("size of label is", label.shape)
            # print(label)
            # print("size of mask is", mask.shape)
            return (img, label, mask)
        
    def __len__(self):
        return  len(self.img_paths)

def get_loader(batch_size=64, size=600, num_curves=9, noise=0.1, mask=False):
    transforms = Compose([
        RandomCrop(600, 600),
        Resize(size),
        RandomHflip(),
        ToTensor(),
        # RandomNoise(0, noise),
        Normalize()  #TODO: Find mean, var of img and labs
    ])

    ds = EyeSegmentDataset(transforms=transforms, num_curves=num_curves,file_path=False, mask=mask)

    #changes
    l = len(ds)
    print('Length of ds:' , l)
    train_set, val_set = torch.utils.data.random_split(ds, [l-20, 20])
    # print(train_set)
    # print(len(train_set))
    # print(len(val_set))
    
    dl_train = DataLoader(train_set,batch_size, shuffle=True, num_workers=12)

    # print("inside get loader")
    # dataiter = iter(dl_train) 
    # images, labels = dataiter.next() 
    # print(type(images))
    # print(images.shape) 
    # print(labels.shape)

    dl_test = DataLoader(val_set,batch_size, shuffle=True, num_workers=12)


    return dl_train,dl_test
    
def main():
    transforms = Compose([
        RandomCrop(600, 600),
        Resize(300),
        RandomHflip(),
        ToTensor(),
        # RandomNoise(0, 0.05),
        Normalize()  #TODO: Find mean, var of img and labs
    ])

    ds = EyeSegmentDataset(transforms=transforms, num_curves=9, mask=True)
    # ds[0]
    
    ## Code bellow i.) find the counts of images under each size: {920: 225, 1010: 450, 919: 75}
    ## ii.) Finds Maximum width of the labels: except 2 (79, 749)  all are under 500
    ## iii.) If width is 0, then removes the image
    # size_map = {}
    # lab_w = []
    # for i in range(len(ds)):
    #     img, lab, fp = ds[i]
    #     lab_w.append((lab.max()-lab.min()).item())
    #     if lab_w[-1] ==0:
    #         print(i, fp)
    #         os.remove(fp)

    #         fig = layer_plot(img[0], lab)
    #         fig.savefig("test"+str(i)+".png")
    
    #     keys = list(size_map.keys())
    #     if img.shape[1] not in keys:
    #         size_map[img.shape[1]] = 0
    #     size_map[img.shape[1]]+= 1
    # print(size_map)
    # print(max(lab_w), lab_w)

    # Save single image
    # img, lab = ds[74]
    # print(lab)
    # fig = layer_plot(img[0], lab)
    # fig.savefig('test.png')
    # print(len(ds))

    dl = DataLoader(ds,25, shuffle=True)
    imgs, labs, masks = iter(dl).next()
    print(imgs.shape, labs.shape, masks.shape)
    fig = plot_grid(imgs, labs, masks)
    fig.savefig('grid.png', bbox_inches='tight')
    
if __name__ == "__main__":
    main()
