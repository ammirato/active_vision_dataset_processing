import os
import numpy as np
import sys
import json
import cv2
import torch
import collections

images_dir = 'jpg_rgb'
annotation_filename = 'annotations.json'


def collate(batch):
    images = []
    labels = []
    for img,label in batch:
        images.append(img)
        labels.append(label)
    if isinstance(labels[0],int):#classification id labels
        labels = torch.from_numpy(np.array(labels))

    if len(images) == 1:
        images = images[0]
        labels = labels[0]

    #images = torch.stack(images)
    
    return [images,labels]


class AVD(object):
    """
    Organizes data from the Active Vision Dataset

    Different from AVD in that this class returns only a single 
    image and bounding box per index, for classification or detection.
    All boxes are still present, they will just require multiple indexes
    to access. 

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
                 scene_list=None, classification=False, preload_images=False,
                 by_box=False, class_id_to_name=None, fraction_of_no_box=1):
        """
        Create instance of AVDd class

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
                                 according to labeled box, after target 
                                 transform. Then transforms labels to just
                                 id (no box). If not 'by_box' each image becomes
                                 a list of numpy arrays, one for each instance. 
          preload_images(False): if images should all be loaded 
                                   during initialization
          by_box(False): if data will be returned by box
                           i.e. returns only a single 
                           image and bounding box per index, 
                           for classification or detection.
                           All boxes are still present, they will 
                           just require multiple indexes to access.
          class_id_to_name(None): dict with keys=class ids, values = names
                                  Assumes original class ids, any changes
                                  to ids via a target transform will 
                                  be applied by this object.  
          fraction_of_no_box(float=1): fraction of images without a ground
                                     truth box to keep(1 keeps all of them)
                 
        """

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classification = classification 
        self.train = train # training set or test set
        self.class_id_to_name = class_id_to_name 
        self.fraction_of_no_box = fraction_of_no_box       
 
        #if no inputted scene list, use defaults 
        if scene_list == None:
            if self.train:
                scene_list = self.default_train_list
            else:
                scene_list = self.default_test_list
        self.scene_list = scene_list

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

      
        #get all the image names, and annotations 
        image_names = []
        name_to_index = {}
        index = 0
        init_targets = {}
        init_navs = {}
        for scene in self.scene_list:
            cur_image_names = (os.listdir(os.path.join(self.root,   
                                                      scene,
                                                      images_dir))) 
            #sort them so they are in a know order, scenes stay together
            cur_image_names.sort()
            used_image_names = []
            #image_names.extend(cur_image_names)

            #get annotations
            with open(os.path.join(self.root,scene,annotation_filename)) as f:
                annotations = json.load(f)
            for name in cur_image_names:  
                target = annotations[name]['bounding_boxes']
                transformed_target = self.target_transform(list(target))
                if  transformed_target == [] or (len(transformed_target) == 1 and transformed_target[0][4] == 0):
                    if np.random.rand() > self.fraction_of_no_box:
                        continue 
                init_targets[name] = list(target)
                cur_navs = annotations[name] 
                cur_navs.pop('bounding_boxes')
                init_navs[name] = cur_navs
                name_to_index[name] = index
                index += 1 
                used_image_names.append(name)

            image_names.extend(used_image_names)

        self.image_names = image_names
        self.init_targets = init_targets 
        self.init_navs = init_navs        
        self.name_to_index = name_to_index
 
        self.preload_images = preload_images 
        if preload_images:
            self.__preload_images__()

        self.by_box = by_box
        if by_box:
            self.__set_name_and_box_index_list__()

        #set id_to_name dict to fit the transformed target ids
        if self.class_id_to_name is not None:
            self.transform_id_to_name_dict()






    def __set_name_and_box_index_list__(self):
        name_and_box_index = []

        for name in self.image_names:
            target = self.init_targets[name]
            if self.target_transform is not None:
                target = self.target_transform(target)
            #if there is no gt box, add a background one
            if len(target) == 0:
                target = np.asarray([[0,0,100,100,0,0]])
                self.init_targets[name] = target


            #for each box, record name and index in target 
            index_counter = 0
            for box in target:
                assert(index_counter < len(target))
                name_and_box_index.append([name,index_counter])
                index_counter+=1

        self.name_and_box_index = name_and_box_index


#    def __set_name_and_box_index_list__(self):
#        image_names = []
#        name_and_box_index = []
#        init_targets = {}        
#
#        for scene in self.scene_list:
#            #get image names for this scene
#            cur_image_names = os.listdir(os.path.join(self.root,   
#                                                      scene,
#                                                      images_dir))
#            #sort them so they are in a known order
#            cur_image_names.sort() 
#
#            #get annotations for this scene
#            with open(os.path.join(self.root,scene,annotation_filename)) as f:
#                annotations = json.load(f)
#       
#            #for each image, get the boxes desired 
#            for name in cur_image_names:  
#                target = annotations[name]['bounding_boxes']  
#                init_targets[name] = list(target)
#                if self.target_transform is not None:
#                    target = self.target_transform(target)
#                #skip some fraction of images without a gt box
#                if len(target) == 0:
#                    if np.random.rand() > self.fraction_of_no_box:
#                        continue    
#                    else:
#                        target = np.asarray([[0,0,100,100,0,0]])
#                        init_targets[name] = target
#
#  
#                #for each box, record name and index in target 
#                index_counter = 0
#                for box in target:
#                    assert(index_counter < len(target))
#                    name_and_box_index.append([name,index_counter])
#                    index_counter+=1
#            image_names.extend(cur_image_names) 
#
#        self.name_and_box_index = name_and_box_index
#        self.image_names = image_names      
#        self.init_targets = init_targets



    def __preload_images__(self):

        images = {} 
        for scene in self.scene_list:
            #get image names# for this scene
            cur_image_names = os.listdir(os.path.join(self.root,   
                                                      scene,
                                                      images_dir))
            #sort them so they are in a known order
            cur_image_names.sort() 

            for image_name in cur_image_names:  
                if image_name[0] == '0':
                    scene_type = 'Home'
                elif image_name[0] == '1':
                    scene_type = 'Office'
                elif image_name[0] == '2':
                    scene_type = 'Gen'
                scene_name = scene_type + "_" + image_name[1:4] + "_" + image_name[4]
            
                #read the image 
                img = cv2.imread(os.path.join(self.root,scene_name, 
                                               images_dir,image_name))
                
                images[image_name] = img

                if len(images) %50 ==0:
                    print(len(images))

        self.preload_images = True
        self.images = images 




    def __getitem__(self, index):
        """ 
        Gets desired image and label   
        """
        if not self.by_box:
            #get the image name and box
            image_name = self.image_names[index]

            #name_and_index needs to be alist of lists
            #if(len(image_name) >0 and type(image_name[0]) is not list): 
            if(len(image_name) >0 and type(image_name) is not list): 
                image_name = [image_name]        
     
            image_target_list = []
            image_list = []
            target_list = []

            for name in image_name:
                #build the path to the image and annotation file
                #see format tab on Get Data page on AVD dataset website
                if name[0] == '0':
                    scene_type = 'Home'
                elif name[0] == '1':
                    scene_type = 'Office'
                elif name[0] == '2':
                    scene_type = 'Gen'
                scene_name = scene_type + "_" + name[1:4] + "_" + name[4]
            
                #read the image and bounding boxes for this image
                #(doesn't get the movement pointers) 
                if self.preload_images:
                    img = self.images[name]
                else:
                    img = cv2.imread(os.path.join(self.root,scene_name, 
                                                   images_dir,name))
                

                #get the target and apply transform
                target = list(self.init_targets[name])
                if self.target_transform is not None:
                    target = self.target_transform(target)


                #crop images for classification if flag is set
                if self.classification:
                    images = []
                    ids = []
                    for box in target:
                        try:
                            cur_img = img[box[1]:box[3],box[0]:box[2],:]
                        except:
                            print(name)
                            print(box)
                            print(self.root)
                            print(scene_name)
                            print(images_dir)
                            print(img)

                        if self.transform is not None:
                            cur_img = self.transform(cur_img)
                        images.append(cur_img)
                        ids.append(box[4])
                        

                    img = images
                    target = ids
     
                #apply image transform if  not classification     
                elif self.transform is not None:
                    img = self.transform(img)

                #add image name to targets
                target = [target, name]

                #add navigation pointers if not classification
                if not self.classification:
                    target.append(self.init_navs[name])

                image_target_list.append([img,target])
                image_list.append(img)
                target_list.append(target)


            #special case for single image/label
            if(len(image_target_list) == 1):
                image_target_list = image_target_list[0]
                image_list = image_list[0]
                target_list = target_list[0]

            #ureturn image_target_list
            if len(image_list) == 0:
                print(image_name)
                print(self.root)
                print(scene_name)
                print(images_dir)

            return [image_list,target_list]


        # #######################33 
        # by box 
        ########################3 
        else:

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
                elif image_name[0] == '1':
                    scene_type = 'Office'
                elif image_name[0] == '2':
                    scene_type = 'Gen'
                scene_name = scene_type + "_" + image_name[1:4] + "_" + image_name[4]
            
                #read the image and bounding boxes for this image
                #(doesn't get the movement pointers) 
                if self.preload_images:
                    img = self.images[image_name]
                else:
                    img = cv2.imread(os.path.join(self.root,scene_name, 
                                                   images_dir,image_name))
                

                #get the target and apply transform
                target = list(self.init_targets[image_name])
                if self.target_transform is not None:
                    target = self.target_transform(target)
               
                #get the single box
                target = target[box_index]


                #crop images for classification if flag is set
                if self.classification:
                    #img = np.asarray(img)
                    img = img[target[1]:target[3],target[0]:target[2],:]
                    target = target[4] 

                #apply image transform     
                if self.transform is not None:
                    img = self.transform(img)
                
                #add image name to targets
                target = [np.expand_dims(target,axis=0), image_name]

                #add navigation pointers if not classification
                if not self.classification:
                    target.append(self.init_navs[image_name])

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
        if not self.by_box:
            return len(self.image_names) 
        else:
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



    def get_count_by_class(self):
        """
        Returns a count of how many labels there are per class

        Assumes class id is still 5th element of each target
        even after target transform
        """
        #if this was already computed, don't do it again
        if hasattr(self,'count_by_class'):
            return self.count_by_class

        #count_by_class = np.zeros(num_classes) 
        count_by_class = {} 
         
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
              
                #for each box, update class count
                for kl in range(len(target)):
                    class_id = target[kl][4]
                    if class_id not in count_by_class.keys():
                        count_by_class[class_id] = 0 
                    count_by_class[class_id] += 1 

        #save for later and return
        self.count_by_class = count_by_class    
        return self.count_by_class 


    def get_num_classes(self):
        #return len(self.get_count_by_class().keys()) 
        return len(self.class_id_to_name.keys()) 


    def get_name_index(self,name):
        """
        Get index of an image by name.

        Only works for detection, not by box
        """
        if self.classification or self.by_box:
            return None
        return self.name_to_index[name]



    def get_class_names(self):
        return self.class_id_to_name.values()
        

    def transform_id_to_name_dict(self):
        """
        Changes id->name to reflect changes made to ids from target transform
        """
        
        #get a list of ids after transform
        self.get_count_by_class()
        ids_after = self.count_by_class.keys()

        #for each original id, make a dummy box, transform it,
        #and make a new key,value in a new dict with new_id,name
        new_dict = {}
        ids_before = self.class_id_to_name.keys()
        for old_id in ids_before:
            dummy_box = [0,0,0,0,old_id,0]
            transformed_box = self.target_transform([dummy_box])
            if len(transformed_box) > 1:
                transformed_box = [transformed_box[-1]]
            #check to see if box is None of empty
            if transformed_box is None or not transformed_box:   
                continue
            if sum(transformed_box[0][0:4]) > 0:
                continue
            new_id = transformed_box[0][4]
            new_dict[new_id] = self.class_id_to_name[old_id]
     

        self.class_id_to_name = new_dict

        return None


    def get_original_bboxes(self):
        """
        Returns dict of image_name->untransformed bounding boxes
        """
        return self.init_targets



    def get_box_difficulty(self,box):
        """
        Returns box difficulty measure, as defined on dataset website
        """
        box_dims = np.array([box[2]-box[0], box[3]-box[1]])
        maxd = box_dims.max()
        mind = box_dims.min()

        if maxd>=300 and mind>=100:
            return 1
        elif maxd>=200 and mind>=75:
            return 2
        elif maxd>=100 and mind>=50:
            return 3 
        elif maxd>=50 and mind>=30:
            return 4 
        else:
            return 5




