from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import json
import torch

images_dir = 'jpg_rgb'
annotation_filename = 'annotations.json'
class AVD(data.Dataset):
  """
  Organizes data from the Active Vision Dataset

  Uses design from pytorch, torchvision, to provide indexable 
  data structure that returns images and labels from the dataset. 

  This class should be used as a starting point for those wishing to
  learn how to access our data. Currently it is not for actual use as is,
  it must be modified to give the desired output.
  """

  #these are the train/test split 1 used in our original paper
  default_train_list = [
                'Home_002_1',
                'Home_003_1',
                'Home_003_2',
                'Home_004_1',
                'Home_004_2',
                'Home_005_1',
                'Home_005_2',
                'Home_006_1',
                'Home_014_1',
                'Home_014_2',
                'Office_001_1'

  ]
  default_test_list = [
                'Home_001_1',
                'Home_001_2',
                'Home_008_1'
  ]




  def __init__(self, root, train=True, transform=None, target_transform=None, scene_list=None):
    """
      Creat instance of AVD class

      Ex) traindata = AVD('/path/to/data/')

      INPUTS:
        root: root directory of all scene folders

      KEYWORD INPUTS:
        train(true): whether to use train or test data(only has an effect if scene_list==None)
        transform(None): function to apply to images before returning them(i.e. normalization)
        target_transform(None): function to apply to labels before returning them
        scene_list(None): which scenes to get images/labels from 
     
    """

    self.root = root
    self.transform = transform
    self.target_transform = target_transform
    self.train = train # training set or test set
    
    #if no inputted scene list, use defaults 
    if scene_list == None:
      print('Using default scene list...')
      if self.train:
        scene_list = self.default_train_list
      else:
        scene_list = self.default_test_list
    self.scene_list = scene_list

    if not self._check_integrity():
      raise RuntimeError('Dataset not found or corrupted.')

    #get all the image names
    image_names = []
    for scene in self.scene_list:
      image_names.extend(os.listdir(os.path.join(self.root, scene, images_dir))) 
    self.image_names = image_names
    #sort them so they are in a know order, scenes stay together
    self.image_names.sort()






  def __getitem__(self, index):
    """ Gets desired image and label  """

    #get the image name 
    image_name = self.image_names[index]

    #build the path to the image and annotation file
    #see format tab on Get Data page on AVD dataset website
    if image_name[0] == '0':
      scene_type = 'Home'
    else:
      scene_type = 'Office'
    scene_name = scene_type + "_" + image_name[1:4] + "_" + image_name[4]
  
    #read the image and bounding boxes for this image
    #(doesn't get the movement pointers) 
    img = Image.open(os.path.join(self.root,scene_name,images_dir,image_name))
    with open(os.path.join(self.root,scene_name,annotation_filename)) as f:
      annotations = json.load(f)
    target = annotations[image_name]['bounding_boxes']    
    
    #apply the transforms           
    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      target = self.target_transform(target)

    #HACK ATTACK
    #
    #Here is where things get fishy due to pytorch's DataLoader
    #Currently I cannot figure out how to return variable size 
    #arrays for the labels, so every image just gets one box
    #and gets a background box if it doesn't have any objects
    #
    #if there are not bounding boxes in this image, 
    #make a background box
    if len(target) == 0:
      target = [[100, 100, 500,500,0,1]]
    #only output first box because  idk
    target = torch.Tensor(np.array(target[0]))  


    return img, target




  def __len__(self):
    """ Gives number of images"""
    return len(self.image_names) 

  def _check_integrity(self):
    """ Checks to see if all scenes in self.scene_list exist

        Checks for existence of root/scene_name, root/scene_name/jpg_rgb,
        root/scene_name/annotations.json
    """
    root = self.root
    for scene_name in self.scene_list:
      if not(os.path.isdir(os.path.join(root,scene_name)) and 
           os.path.isdir(os.path.join(root,scene_name, images_dir)) and
           os.path.isfile(os.path.join(root,scene_name,annotation_filename))):
        return False
    return True



