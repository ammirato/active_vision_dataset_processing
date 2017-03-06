import active_vision_dataset
import transforms
import numpy as np


#USE 1
#basic use for getting images/labels for detection
trainset = active_vision_dataset.AVD(root='/playpen/ammirato/Data/RohitData')
                                    
#get an image and its label(s) 
image,labels = trainset[0]

print('USE 1: Number of boxes: ' + str(len(labels)))
print('USE 1: First box: ' + str(labels[0][0:4]))
print('USE 1: First box instance id: ' + str(labels[0][4]))
print('USE 1: First box difficulty: ' + str(labels[0][5]))



#USE 2 
#using custom scene list, applying a transform

#use images/labels from these scenes only
#MUST be a list, even for a single scene!
scene_list = ['Home_001_1', 'Home_002_1', 'Home_003_1']

#get labels from only the first 25 instances
target_trans = transforms.PickInstances(range(25))

trainset = active_vision_dataset.AVD(root='/playpen/ammirato/Data/RohitData', 
                                    scene_list=scene_list,
                                    target_transform=target_trans)

#get an image and its label(s) 
image,labels = trainset[0]
#get the next image and its label(s)
image2,labels2 = trainset[1]

#get a batch of images/labels
batch_images_labels = trainset[0:10]
#get the first image/label pair in the batch
batch_image1, batch_label1 = batch_images_labels[0]
#get all the images in the batch
all_images_in_batch = [img for img,_ in batch_images_labels]


#5th element(index 4) of each box is the instance id,
#so here we print out the max index id over all boxes
print('\nUSE 2: Max instance id (will be <25): ' + str(max(np.asarray(labels2)[:,4])))




#USE 3
#crop images around boxes for classification, 
#add some perturbed boxes, validate the boxes

#add more boxes to get more data
perturb_trans = transforms.AddPerturbedBoxes()#can add custom perturbations
#add background boxes
back_trans = transforms.AddBackgroundBoxes()
#make sure boxes are valid 
validate_trans = transforms.ValidateMinMaxBoxes()
#compose the three transformations, peturbing then add bg then validate 
target_trans = transforms.Compose([perturb_trans,back_trans,validate_trans])

trainset = active_vision_dataset.AVD(root='/playpen/ammirato/Data/RohitData', 
                                    target_transform=target_trans,
                                    classification=True)

#get a list of images and their labels 
images,labels = trainset[0]

image1 = images[0]
image2 = images[1]


print('\nUSE 3: First classification images instance id: ' + str(labels[0]))






#USE 4 
#Get one box at a time, detection
trainset = active_vision_dataset.AVD_ByBox(root='/playpen/ammirato/Data/RohitData')


image,label = trainset[0] #gives first image, first box
image,label = trainset[1] #gives either:
                          #        first image, second box (if their is a second box)
                          #        second image, first box (otherwise)


#above comments assume first image and second image have at least one labeled box
#if an image does not have a box, it will be "skipped" by this data structure.
#so if the first image does not have a box but the second does, then
#trainset[0] would return second image, first box


print('\nUSE 4 label: ' + str(label))





#USE 5 
#Get one box at a time, classification, pick instances 
image,label = trainset[0] #gives first image, first box
image,label = trainset[1] #gives either:
                          #        first image, second box (if their is a second box)
                          #        second image, first box (otherwise)


#above comments assume first image and second image have at least one labeled box
#if an image does not have a box, it will be "skipped" by this data structure.
#so if the first image does not have a box but the second does, then
#trainset[0] would return second image, first box
 
 
#get labels from only instance 5, 7, 23 
target_trans = transforms.PickInstances([5,7,23])

trainset = active_vision_dataset.AVD_ByBox(root='/playpen/ammirato/Data/RohitData',
                                           target_transform=target_trans,
                                            classification=True)
image,label = trainset[0]

print('\nUSE 5 label: ' + str(label))
#really you could leave classification flag off and crop yourself



