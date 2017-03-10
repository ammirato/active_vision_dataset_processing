import os
import numpy as np
import sys
import json
import cv2
import torch
import collections

images_dir = 'jpg_rgb'
annotation_filename = 'annotations.json'


#http://pytorch.org/docs/_modules/torch/utils/data/dataloader.html#DataLoader
def collate(batch):
    if isinstance(batch[0],torch.LongTensor):#detection box labels
        return batch 
    elif isinstance(batch[0],int):#classification id labels
        #batch = [torch.from_numpy(np.array([el])) for el in batch]
        #return torch.stack(batch)
        return torch.from_numpy(np.array(batch))
    elif isinstance(batch[0],torch.FloatTensor):#images
        return torch.stack(batch)
    elif isinstance(batch[0], collections.Iterable):#list[images,labels]
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed] 







class AVD():
    """
    Organizes data from the Active Vision Dataset

    Uses design from pytorch, torchvision, to provide indexable 
    data structure that returns images and labels from the dataset. 
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




    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 scene_list=None, classification=False):
        """
        Creat instance of AVD class

        Ex) traindata = AVD('/path/to/data/')

        INPUTS:
          root: root directory of all scene folders

        KEYWORD INPUTS(default value):
          train(true): whether to use train or test data
                       (only has an effect if scene_list==None)
          transform(None): function to apply to images before 
                           returning them(i.e. normalization)
          target_transform(None): function to apply to labels 
                                  before returning them
          scene_list(None): which scenes to get images/labels from,
                            if None use default lists 
          classification(False): whether to use cropped images for 
                                 classification. Will crop each box 
                                 according to labeled box. Then transforms
                                 labels to just id (no box). Each image becomes
                                 a list of numpy arrays, one for each instance. 
                                 Be careful using a target_transform with this 
                                 on, targets must remain an array with one row
                                 per box.
        """

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classification = classification 
        self.train = train # training set or test set
        
        #if no inputted scene list, use defaults 
        if scene_list == None:
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
            image_names.extend(os.listdir(os.path.join(self.root,   
                                                      scene,
                                                      images_dir))) 
        self.image_names = image_names
        #sort them so they are in a know order, scenes stay together
        self.image_names.sort()




    def __getitem__(self, index):
        """ Gets desired image and label    """

        #get the image name 
        image_names = self.image_names[index]

        #make single name a list
        if(type(image_names) is not list):
            image_names = [image_names]

        image_target_list = []
        image_list = []
        target_list = []
        for image_name in image_names:

            #build the path to the image and annotation file
            #see format tab on Get Data page on AVD dataset website
            if image_name[0] == '0':
                scene_type = 'Home'
            else:
                scene_type = 'Office'
            scene_name = scene_type + "_" + image_name[1:4] + "_" + image_name[4]
        
            #read the image and bounding boxes for this image
            #(doesn't get the movement pointers) 
            img = cv2.imread(os.path.join(self.root,scene_name, 
                                           images_dir,image_name))
            with open(os.path.join(self.root,scene_name,annotation_filename)) as f:
                annotations = json.load(f)
            target = annotations[image_name]['bounding_boxes']        
            
            #apply target transform
            if self.target_transform is not None:
                target = self.target_transform(target)

            #crop images for classification if flag is set
            if self.classification:
                img = np.asarray(img)
                images = []
                ids = []
                for box in target:
                    cur_img = img[box[1]:box[3],box[0]:box[2],:]

                    if self.transform is not None:
                        cur_img = self.transform(cur_img)
                    images.append(cur_img)
                    ids.append(box[4])

                img = images
                target = ids
            #apply image transform if not classification  
            elif self.transform is not None:
                img = self.transform(img)

            image_list.append(img)
            target_list.append(target)
            image_target_list.append([img,target])

        #TODO - remove this special case
        #special case for single image/label
        #if( not(self.classification) and len(image_target_list) == 1):
        if(len(image_target_list) == 1):
            image_target_list = image_target_list[0]
            image_list = image_list[0]
            target_list = target_list[0]

        #return image_target_list
        return [image_list,target_list]




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








class AVD_ByBox():
    """
    Organizes data from the Active Vision Dataset

    Different from AVD in that this class returns only a single 
    image and bounding box per index, for classification or detection
   
    All boxes are still present, they will just require multiple indexes
    to access. 

    """
    #TODO - too much copy/paste with AVD


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




    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 scene_list=None, classification=False):
        """
        Create instance of AVD_OnebyOne class

        Ex) traindata = AVD('/path/to/data/')

        INPUTS:
          root: root directory of all scene folders

        KEYWORD INPUTS(default value):
          train(true): whether to use train or test data
                       (only has an effect if scene_list==None)
          transform(None): function to apply to images before 
                           returning them(i.e. normalization)
          target_transform(None): function to apply to labels 
                                  before returning them
          scene_list(None): which scenes to get images/labels from,
                            if None use default lists 
          classification(False): whether to use cropped images for 
                                 classification. Will crop each box 
                                 according to labeled box. Then transforms
                                 labels to just id (no box). Each image becomes
                                 a list of numpy arrays, one for each instance. 
                                 Be careful using a target_transform with this 
                                 on, targets must remain an array with one row
                                 per box.
        """

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classification = classification 
        self.train = train # training set or test set
        
        #if no inputted scene list, use defaults 
        if scene_list == None:
            if self.train:
                scene_list = self.default_train_list
            else:
                scene_list = self.default_test_list
        self.scene_list = scene_list

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

        #get all the image names
        image_names = []
        name_and_box_index = []
        for scene in self.scene_list:
            #get image names for this scene
            cur_image_names = os.listdir(os.path.join(self.root,   
                                                      scene,
                                                      images_dir))
            #sort them so they are in a known order
            cur_image_names.sort() 

            #get annotations for this scene
            with open(os.path.join(self.root,scene,annotation_filename)) as f:
                annotations = json.load(f)
       
            #for each image, get the boxes desired 
            for name in cur_image_names:  
                target = annotations[name]['bounding_boxes']  
                if self.target_transform is not None:
                    target = self.target_transform(target)
              
                #for each box, record name and box
                for kl in range(len(target)):
                    name_and_box_index.append([name,kl])
                     
            image_names.extend(cur_image_names) 



        self.image_names = image_names
        self.name_and_box_index = name_and_box_index




    def __getitem__(self, index):
        """ 
        Gets desired image and label   
        """

        #get the image name and box
        #image_name,box_index = self.name_and_box_index[index]
        name_and_index = self.name_and_box_index[index]
        #name_and_index needs to be alist of lists
        if(len(name_and_index) >0 and type(name_and_index[0]) is not list): 
            name_and_index = [name_and_index]        
 
        image_target_list = []
        image_list = []
        target_list = []

        for image_name,box_index in name_and_index:
            #build the path to the image and annotation file
            #see format tab on Get Data page on AVD dataset website
            if image_name[0] == '0':
                scene_type = 'Home'
            else:
                scene_type = 'Office'
            scene_name = scene_type + "_" + image_name[1:4] + "_" + image_name[4]
        
            #read the image and bounding boxes for this image
            #(doesn't get the movement pointers) 
            img = cv2.imread(os.path.join(self.root,scene_name, 
                                           images_dir,image_name))
            with open(os.path.join(self.root,scene_name,annotation_filename)) as f:
                annotations = json.load(f)
            target = annotations[image_name]['bounding_boxes']        
            
            #apply target transform
            if self.target_transform is not None:
                target = self.target_transform(target)

            #get the single box
            target = target[box_index]

            #crop images for classification if flag is set
            if self.classification:
                img = np.asarray(img)
                img = img[target[1]:target[3],target[0]:target[2],:]
                target = target[4] 
        
           
            #apply image transform     
            if self.transform is not None:
                img = self.transform(img)

            image_target_list.append([img,target])
            image_list.append(img)
            target_list.append(target)


        #special case for single image/label
        if(len(image_target_list) == 1):
            image_target_list = image_target_list[0]
            image_list = image_list[0]
            target_list = target_list[0]

        #ureturn image_target_list
        return [image_list,target_list]




    def __len__(self):
        """ 
        Gives number of boxes
        """
        return len(self.name_and_box_index) 

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
