import torch
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import json
import random
import cv2
import collections

#Below are the transforms that depend on torch
#Delete these to remove the torch dependency
class ToTensor(object):
    """
    Converts PIL image to tensor. Does NOT normalize 

    Copied from 
    https://github.com/pytorch/vision/blob/master/torchvision/transforms.py

    """    

    def __call__(self,in_img):

        if isinstance(in_img,np.ndarray):
            img = np.transpose(in_img,(2,0,1))
            img = torch.from_numpy(img)
        elif isinstance(in_img,list):
            img = np.asarray(in_img)
            img = torch.from_numpy(img)
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(in_img.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if in_img.mode == 'YCbCr':
                nchannel = 3
            else:
                nchannel = len(in_img.mode)
            img = img.view(in_img.size[1], in_img.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous() 
        return img




class ToNumpy(object):
    """
    Converts tensor CxHxW to numpy HxWxC.


    """
    def __call__(self,in_img):

        #remove batch demension
        if len(in_img.size()) == 4:
            in_img = in_img.squeeze(0)

        #make sure tensor is on cpu before converting to numpy
        npimg = in_img.cpu().numpy()
        #make HxWxC
        npimg = np.transpose(npimg,(1,2,0))
        
        return npimg


class BGRToRGB(object):
    """
    Converts BGR numpy image to RGB.


    """
    def __call__(self,in_img):

        b = in_img[:,:,0] 
        g = in_img[:,:,1] 
        r = in_img[:,:,2] 

        return np.stack((r,g,b),axis=2)

class RGBToBGR(object):
    """
    Converts RGB numpy image to BGR.


    """
    def __call__(self,in_img):

        r = in_img[:,:,0] 
        g = in_img[:,:,1] 
        b = in_img[:,:,2] 

        return np.stack((b,g,r),axis=2)


class NormalizePlusMinusOne(object):
    """
    Changes an tensors's values so 0->-1 and 255->1

    Assumes input is a tensor with values in [0,255] 
    """ 

    def __call__(self,tensor):
       return (tensor.float()-127.5)/127.5


class DenormalizePlusMinusOne(object):
    """
    Changes an tensors's values so -1->0 and 1->255

    Assumes input is a tensor with values in [-1,1] 
    """ 

    def __call__(self,tensor):
       return (tensor.float()*127.5) + 127.5


class NormalizeRange(object):
    """
    Changes an tensors's values to be in a given range.

    Assumes original values are [0,255], unless args set
   
    ARGS:
        to_min: new min value  
        to_max: new max value  

    KEYWORD ARGS:
        from_min(Float=0.0): the orignal distributions min value
        from_max(Float=255.0): the orignal distributions max value

    Follows http://stackoverflow.com/questions/5294955/how-to-scale-down-a-range-of-numbers-with-a-known-min-and-max-value 

    """ 
    def __init__(self,to_min,to_max,from_min=0.0,from_max=255.0):
        self.to_min = to_min
        self.to_max = to_max 
        self.from_min = from_min
        self.from_max = from_max

    def __call__(self,tensor):
        scale = (self.to_max-self.to_min)/(self.from_max-self.from_min)
        return scale*(tensor.float() - self.from_min) + self.to_min 


class MeanSTDNormalize(object):
    """
    Normalizes a numpy array values by (array-mean)/std

    Assumes array, mean, std have same number of channels

    KWARGS: 
        mean
        std
    """
    #TODO: add functionality for tensors

    def __init__(self,mean=0,std=1):
        self.mean = mean
        self.std = std

    def __call__(self,array):
           return (array-self.mean)/self.std 

class ToPILImage(object):
    """
    Converts tensor to PIL Image. Does not change any values in tensor.

    Copied from 
    https://github.com/pytorch/vision/blob/master/torchvision/transforms.py
    """

    def __call__(self,tensor):
        npimg = tensor.numpy()
        npimg = np.transpose(npimg,(1,2,0))
        return Image.fromarray(npimg.astype(np.uint8)) 




## END TORCH DEPENDENCY
##############################################################################



#https://github.com/pytorch/vision/blob/master/torchvision/transforms.py
class Compose(object):
      """
      Composes several transforms together.

      Copied from 
      https://github.com/pytorch/vision/blob/master/torchvision/transforms.py 

      Args:
          transforms (List[Transform]): list of transforms to compose.
      Example:
           transforms.Compose([
               transforms.AddPerturbedBoxes(),
               transforms.ValidateMinMaxBoxes(),
           ])
      """

      def __init__(self, transforms):
          self.transforms = transforms

      def __call__(self, data):
          for t in self.transforms:
              data = t(data)
          return data 









class PickInstances(object):
    """
    Removes all instances that are not selected from the target labels
      ARGUMENTS:
        instance_ids (List[ids]): which instances to keep, 
                                  identified by their int id
      ex) transforms.PickInstance([3,5,11,19])
          
    """


    def __init__(self, instance_ids, max_difficulty=None):
        """
        Defines which instances to keep

        ARGUMENTS:
        instance_ids: which instances to keep, identified by their int id

        KEYWORD ARGS:
        max_difficulty (int)=None: pick boxes with smaller difficulty than this

        ex) PickInstance([3,5,11,19])
        """ 
        self.instance_ids = instance_ids
        self.max_difficulty = max_difficulty 

    def __call__(self,target):
        selected_targets = [] 
        for box in target:
            if box[4] in self.instance_ids:
                if (self.max_difficulty is None or 
                                            box[5] <= self.max_difficulty): 
                    selected_targets.append(box)
        return selected_targets
      
     


class AddPerturbedBoxes(object):
    """
    Adds a number of boxes to each image by randomly perturbing the present boxes.

    Changes the first 4 numbers in each box. Recommened to use with some
    box validation, as the pertubations may move boxes outside of image, or make
    invalid boxes

    KEYWORD ARGS:
      num_to_add (int): how many boxes to add PER INSTANCE. 
      changes (List[List[int,int],...]): range of values that can be
                                         added to each box field
      replace (Bool: True): whether or not to replace the original box.
                            if True, the original box is removed

      ex) transforms.AddPerturbedBoxes(num_to_add=3,
                                       changes = [[-20,5],
                                                  [-20,5],
                                                  [-5,20]
                                                  [-5,20]])
          #adds 3 boxes per instance
          #each side of box can move in each direction
          #these are the defaults
    """ 


    def __init__(self,num_to_add=3,
                      changes=[[-20,5],[-20,5],[-5,20],[-5,20]],
                      replace=True,
                      percents=False):
      self.num_to_add = num_to_add;
      self.changes = changes;
      self.replace = replace
      self.percents = percents   
 

    def __call__(self, targets):
      new_targets = []
      for box in targets:
        if not(self.replace):
            new_targets.append(box) #keep the original box
      
        changes = list(self.changes)
        if self.percents:
          width = box[2]- box[0]
          height = box[3] - box[1] 
          changes[0] = [self.changes[0][0]*width,self.changes[0][1]*width]   
          changes[1] = [self.changes[1][0]*height,self.changes[1][1]*height]   
          changes[2] = [self.changes[2][0]*width,self.changes[2][1]*width]   
          changes[3] = [self.changes[3][0]*height,self.changes[3][1]*height]   

          changes = [[int(x),int(y)] for x,y in changes]
 
        for il in range(self.num_to_add):
          perturbed_box = list(box)#make a new box that will be changed

          for jl in range(4):
          #TODO add index range var to allow different box formats
            #jitter each side of the original box
            perturbed_box[jl] = box[jl] + random.randint(changes[jl][0],
                                                         changes[jl][1])   

          new_targets.append(perturbed_box) #add the perturbed box

      return new_targets






class ValidateMinMaxBoxes(object):
    """
    Ensures boxes conform to some crieria.

    Criteria include: being inside image dimensions
                      having some minimum size for width, height 

    **Assumes boxes are list, with first 4 elements = [xmin, ymin, xmax, ymax]
    **Does not ensure min box dimensions if doing so will make box outside image

    KEYWORD ARGS:
      image_dimensions (List[int,int]) = [1920,1080]: width, height of image
      min_box_dimensions (List[int,int] = [5,5]: min width, height of each box
    """

    def __init__(self, image_dimensions=[1920,1080],
                       min_box_dimensions=[5,5]):

      self.image_dimensions = image_dimensions
      self.min_box_dimensions = min_box_dimensions


    def __call__(self,targets):
      #TODO clean up, there must be a simpiler solution, too much copy/paste
      #      not completely correct, could widen more
      new_targets = []
      for box in targets:
        box = list(box)
        #ensure box is inside image
        if(box[0] < 0):
          box[0] = 0
        if(box[1] < 0):
          box[1] = 0 
        if(box[2] >= self.image_dimensions[0]):
          box[2] = self.image_dimensions[0]-1; 
        if(box[3] >= self.image_dimensions[1]):
          box[3] = self.image_dimensions[1]-1; 


        width = box[2]-box[0]
        width_missing = self.min_box_dimensions[0] - width
        height = box[3]-box[1]
        height_missing = self.min_box_dimensions[0] - height
      
        #add in width and height if box is too small 
        if(width_missing > 1):
          #try to move each edge by half the needed width, but must stay in image
          xmin_amount = int(min([box[0],
                                      np.floor(width_missing/2)])) 
          box[0] -= xmin_amount
          xmax_amount = int(min([self.image_dimensions[0] - 1 - box[2],
                                      width_missing-xmin_amount])) 
          box[2] += xmax_amount 
        if(height_missing > 1):
          ymin_amount = int(min([box[1],
                                      np.floor(height_missing/2)])) 
          box[1] -= ymin_amount
          ymax_amount = int(min([self.image_dimensions[1] - 1 -box[3],
                                      height_missing-ymin_amount])) 
          box[3] += ymax_amount 
 
        

        new_targets.append(box)
      return new_targets








class AddBackgroundBoxes(object):
    """
    Adds a number of background boxes to each image.

    Ensures the background boxes do not have any intersection with
    the other object boxes.

    KEYWORD ARGS:
      num_to_add (int) = 2: how many boxes to add PER IMAGE. 
      box_dimensions_range(List[List[int,int]List[int,int]]) = [5,5,300,300]: 
                           boxes [min_width,min_height, max_width, max_height] 
      image_dimensions (List[int,int]) = [1920,1080]: width, height of image

      ex) transforms.AddBackgroundBoxes(num_to_add=2,
                                        box_dimensions_range=[100,100,200,200])

          #adds 2 boxes per image 
          #each box will be between 100x100 and 200x200 
    """ 


    def __init__(self,num_to_add=2,box_dimensions_range=[5,5,300,300],
                 image_dimensions=[1920,1080]):
        self.num_to_add = num_to_add
        self.box_dims = box_dimensions_range
        self.image_dims = image_dimensions
    
    def __call__(self, targets):
        bg_boxes = []
  
        counter = 0 
        while(len(bg_boxes) < self.num_to_add):
            #make a new bg_box
            xmin = random.randint(0,self.image_dims[0]-self.box_dims[0]) 
            ymin = random.randint(0,self.image_dims[1]-self.box_dims[1]) 
            width = random.randint(self.box_dims[0],self.box_dims[2]) 
            height = random.randint(self.box_dims[1],self.box_dims[3])
            xmax = min(self.image_dims[0],xmin+width)
            ymax = min(self.image_dims[1],ymin+height)
            
            
            #check to see if the background box intersects any object box
            good_box = True
            for box in targets:
                if not(xmax<box[0] or ymax<box[1] or xmin>box[2] or ymin>box[3]):
                    good_box = False
                    break

            if good_box:
                #make new box, bg id is 0, give difficulty of 0 for now
                bg_boxes.append([xmin, ymin,xmax,ymax, 0, 0])
                counter = 0

            if counter > 100:
                print 'No background box fits'
                bg_boxes.append([xmin, ymin,xmax,ymax, 0, 0])
                counter = 100
  
        #do not return references to original target 
        all_boxes = bg_boxes 
        for box in targets:
            box = list(box)
            all_boxes.append(box)
        return all_boxes

        #targets.extend(bg_boxes)
        #return targets




class MakeIdsConsecutive(object):
    """
    Maps class ids to be consecutive, starting at 0

    If the used ids are [2,6,19] they will become [0,1,2]
    """

    def __init__(self, original_ids):
        """
        ARGS:
            original_ids (List[int,...]): the original ids used
        """

        self.original_ids = original_ids
        id_map = {} 
        for il in range(len(original_ids)):
           id_map[original_ids[il]] = il 
       
        self.id_map = id_map 

    def __call__(self,targets):
        consecutive_targets = []
        for box in targets:
            box = list(box)
            box[4] = self.id_map[box[4]]
            consecutive_targets.append(box) 

        return consecutive_targets


#        if len(boxes) == 0:
#            return boxes
#        if isinstance(boxes[0], collections.Iterable):
#            for il in range(len(boxes)):
#                boxes[il][4] = self.id_map[boxes[il][4]]
#        else: #assume just one box
#            boxes[il][4] = self.id_map[boxes[il][4]]
#            
#        return boxes 

        









class ResizeImage(object):
    """
    Changes an image's size. input assumed to be PIL or numpy image. 

    Assumes image Width x Height x Channels., outputs numpy image
    Can either:
        -(default) warp image to new size using OpenCV2 resize 
        -scale one side, and fill in missing values with 0.
         Always results in a square image

    """ 
    #TODO - add crop method 
          #add fill value

    def __init__(self,size, method='warp'):
        """
        initialize resize image transform

        ARGS:
            size (List[int,int]) - the desired size,  Width x Height
                                   The number of channels will stay the same
                                   'fill' method only uses first int to make square
        KEYWORD ARGS:
            method (string='warp'): 'warp' - using opencv2's resize function
                                    'fill' - scale one side, and fill with 0's
                                             only uses first num of size to
                                             make a square image
        """
        self.size = size
        self.method = method

    def __call__(self,image):

        if not(isinstance(image,np.ndarray)):
            image = np.asarray(image)

        if self.method == 'warp':
            image = cv2.resize(image,tuple(self.size[0:2]))

        elif self.method == 'fill':
            #first reshape the image so the larger
            #side is the correct size
            img_size = np.asarray(image.shape)[0:2]
            scale = float(self.size[0])/ img_size.max()
            new_size = np.round(scale*img_size)
            #no dimension can be 0
            if(new_size[0] == 0):
                new_size[0] = 1
            if(new_size[1] == 0):
                new_size[1] = 1


            resized_image = cv2.resize(image,(int(new_size[1]),int(new_size[0])))
            #now make an image of all 0's that is the 
            #correct size, and put the resized image inside
            blank_img = np.zeros((self.size[0],self.size[0],image.shape[2]))
            blank_img[0:resized_image.shape[0],
                      0:resized_image.shape[1],
                                            :] = resized_image

            image = blank_img

        return image




class CombineInstances(object):
    """
    Assigns a single label the boxes from a group of instances

    INPUTS:
        id_list(List[int]): list of ids to combine. Boxes for all of these
                            ids will be reassigned the id of the first 
                            id in the list

    """

    def __init__(self, id_list):
        self.id_list = id_list

    def __call__(self,targets):
        new_targets = []
        index = 0
        for box in targets:
            box = list(box)
            if box[4] in self.id_list:
                targets[index][4] = self.id_list[0]
            index +=1
            new_targets.append(box)
        return new_targets





class AddRandomBoxes(object):
    """
    Adds a number of random boxes to each image.


    KEYWORD ARGS:
      num_to_add (int) = 2: how many boxes to add PER IMAGE. 
      box_dimensions_range(List[List[int,int]List[int,int]]) = [5,5,300,300]: 
                           boxes [min_width,min_height, max_width, max_height] 
      image_dimensions (List[int,int]) = [1920,1080]: width, height of image

      ex) transforms.AddRandomBoxes(num_to_add=2,
                                        box_dimensions_range=[100,100,200,200])

          #adds 2 boxes per image 
          #each box will be between 100x100 and 200x200 
    """ 


    def __init__(self,num_to_add=2,box_dimensions_range=[5,5,300,300],
                 image_dimensions=[1920,1080]):
        self.num_to_add = num_to_add
        self.box_dims = box_dimensions_range
        self.image_dims = image_dimensions
    
    def __call__(self, targets):
        bg_boxes = []
   
        while(len(bg_boxes) < self.num_to_add):
            #make a new bg_box
            xmin = random.randint(0,self.image_dims[0]-self.box_dims[0]) 
            ymin = random.randint(0,self.image_dims[1]-self.box_dims[1]) 
            width = random.randint(self.box_dims[0],self.box_dims[2]) 
            height = random.randint(self.box_dims[1],self.box_dims[3])
            xmax = min(self.image_dims[0],xmin+width)
            ymax = min(self.image_dims[1],ymin+height)
            
            
            #make new box, id is -1, give difficulty of 0 for now
            bg_boxes.append([xmin, ymin,xmax,ymax, -1, 0])


        #do not return references to original target 
        all_boxes = bg_boxes 
        for box in targets:
            box = list(box)
            all_boxes.append(box)
        return all_boxes

        #targets.extend(bg_boxes)
        #return targets

