from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
import json
import random
import cv2






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


    def __init__(self, instance_ids):
      """
      Defines which instances to keep

      ARGUMENTS:
        instance_ids: which instances to keep, identified by their int id

      ex) PickInstance([3,5,11,19])
      """ 
      self.instance_ids = instance_ids


    def __call__(self,target):
      selected_targets = [] 
      for box in target:
        if box[4] in self.instance_ids:
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

      ex) transforms.AddPerturbedBoxes(num_to_add=3,
                                       changes = [[-20,10],
                                                  [-20,10],
                                                  [-10,20]
                                                  [-10,20]])
          #adds 3 boxes per instance
          #each side of box can move up to 10 pixel in each direction
          #these are the defaults
    """ 


    def __init__(self,num_to_add=3,changes=[[-20,10],[-20,10],[-10,20],[-10,20]]):
      self.num_to_add = num_to_add;
      self.changes = changes;

    
    def __call__(self, targets):
      new_targets = []
      for box in targets:
        new_targets.append(box) #keep the original box
        
        for il in range(self.num_to_add):
          perturbed_box = list(box)#make a new box that will be changed

          for jl in range(4):
          #TODO add index range var to allow different box formats
            #jitter each side of the original box
            perturbed_box[jl] = box[jl] + random.randint(self.changes[jl][0],
                                                        self.changes[jl][1])   

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

        targets.extend(bg_boxes)
        return targets






class NormalizePlusMinusOne(object):
    """
    Changes an image's values so 0->-1 and 255->1

    Assumes image is a numpy array Width x Height x Channels 
    """ 

    def __init__(self):
        self.min = -1
        self.max = 1 

    def __call__(self,image):
       return (image-127.5)/127.5


class DenormalizePlusMinusOne(object):
    """
    Changes an image's values so -1->0 and 1->255

    Assumes image is a numpy array Width x Height x Channels, with
    min=-1 and max=1
    """ 

    def __init__(self):
        self.min = -1
        self.max = 1 
    

    def __call__(self,image):
       return (image*127.5) + 127.5




class ResizeImage(object):
    """
    Changes an image's size. 

    Assumes image is a numpy array Width x Height x Channels.
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
        if self.method == 'warp':
            image = cv2.resize(image,self.size)
        elif self.method == 'fill':
            #first reshape the image so the larger
            #side is the correct size
            img_size = np.asarray(image.shape)[0:2]
            scale = float(self.size[0])/ img_size.max()
            new_size = scale*img_size
            resized_image = cv2.resize(image,(int(new_size[1]),int(new_size[0])))
            #now make an image of all 0's that is the 
            #correct size, and put the resized image inside
            blank_img = np.zeros((self.size[0],self.size[0],image.shape[2]))
            blank_img[0:resized_image.shape[0],
                      0:resized_image.shape[1],
                                            :] = resized_image

            image = blank_img

        return image






