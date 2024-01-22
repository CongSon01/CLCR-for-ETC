import pandas as pd
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
import torch


class CustomDataset(VisionDataset):
    def __init__(self, data_list, transform=None, target_transform=None):
        super(CustomDataset, self).__init__(root=None, transform=transform, target_transform=target_transform)

        # targets = []
        # data = []

        # for entry in data_list:
        #     data.append(entry['x'])
        #     targets.append(entry['y'])

        # data = np.array(data)
        # targets = np.array(targets)

        self.data = data_list['x']
        self.targets = data_list['y']

    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]

        # Assuming that each entry in 'x' is a list of pixel values
        # image = Image.fromarray(np.array(image) * 255).convert('RGB')
        # image = Image.fromarray(image)

        # if self.transform is not None:
        #     image = self.transform(image)
        image = torch.Tensor(image)

        return index, image, label

    def __len__(self):
        return len(self.data)
     
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