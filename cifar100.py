from torchvision.datasets import VisionDataset
from sklearn import preprocessing
from PIL import Image

import os
import os.path
import sys

import numpy as np
import pickle

import PIL.Image as Image
from torchvision.transforms import ToTensor, ToPILImage
import pylab
from utils import download_and_extract_archive



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Cifar100(VisionDataset):
    def __init__(self,root, train=True, transform=None, target_transform=None):
        super(Cifar100, self).__init__(root, transform=transform, target_transform=target_transform)

        base_folder = 'cifar-100-python'
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        filename = "cifar-100-python.tar.gz"
        tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
        train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],    
        ]
        test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
        ]
        meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
        }

        targets=[]
        data =[]
        self.train = train 

        download_and_extract_archive(url,self.root, filename=filename, md5=tgz_md5)
        
        if self.train:
            downloaded_list = train_list
        else:
            downloaded_list = test_list

        for file_name, checksum in downloaded_list:
          file_path = os.path.join(self.root,base_folder, file_name)

          with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            #img = pil_loader(file_path)
            data.extend(entry['data'])

            data = np.vstack(data).reshape(-1, 3, 32, 32)
            data = data.transpose((0, 2, 3, 1))

            if 'labels' in entry:
              targets.extend(entry['labels'])
            else:
              targets.extend(entry['fine_labels'])
            
        self.data= data
        self.targets= targets


    def __getitem__(self, index):

            image = self.data[index]
            label = self.targets[index]

            image = Image.fromarray(image)

            if self.transform is not None:
              image = self.transform(image)

            return index,image, label

    def __len__(self):
        
      length = len(self.data)
      return length
     
    def __get_class_images__(self, label):
         class_images = []
         for i in range(0, self.__len__()):
            if self.targets[i] == label:
               class_images.append(i)
         return class_images 

    def __get_number_of_classes__(self):
       length = len(set(self.targets))
       return length

    def __train_data_indexes__(self, proportion):
       images_subset = []
       for i in range(0,self.__get_number_of_classes__()):
         class_subset = self.__get_class_images__(i)
         splitted_class_subset = class_subset[0:int(proportion*len(self.__get_class_images__(i)))]
         images_subset = images_subset+splitted_class_subset
       return images_subset

    def __val_data_indexes__(self, proportion):
       images_subset = []
       for i in range(0,self.__get_number_of_classes__()):
         class_subset = self.__get_class_images__(i)
         splitted_class_subset = class_subset[int(proportion*len(self.__get_class_images__(i))) : len(self.__get_class_images__(i))]
         images_subset = images_subset+splitted_class_subset
       return images_subset
  
    
    def __incremental_train_indexes__(self,proportion):
      class_subset = []
      n = 0
      for j in range(0,10):
        temp = []
        for k in range(n,n+10):  
          var = self.__get_class_images__(k)
          var = var[0:int(proportion*len(self.__get_class_images__(k)))]
          temp = temp + var
        n=n+10
        class_subset.append(temp)
      return class_subset



    def __incremental_val_indexes__(self,proportion):
      class_subset = []
      n = 0
      temp = []
      for j in range(0,10):
        for k in range(n,n+10):  
          var = self.__get_class_images__(k)
          var = var[int(proportion*len(self.__get_class_images__(k))) : len(self.__get_class_images__(k))]
          temp = temp + var
        n=n+10
        class_subset.append(temp)
      return class_subset 



    def _shuffle_(self,vettore):
        indice_vec = []

        for x in range(0,len(vettore)):
            indice = self.__get_class_images__(x)
            indice_vec.append(indice)
        
        for dim in range(0,len(indice_vec)):
            for y in indice_vec[dim]: 
                self.targets[y] = vettore[dim]

