import active_vision_dataset
import transforms



#USE 1
#basic use for getting images/labels for detection
trainset = active_vision_dataset.AVD(root='/playpen/ammirato/Data/RohitData')
                                    
#get an image and its label(s) 
image,labels = trainset[0]

print('Number of boxes: ' + str(len(labels)))
print('First box: ' + str(labels[0][0:4]))
print('First box instance id: ' + str(labels[0][4]))
print('First box difficulty: ' + str(labels[0][5]))





#USE 2 
#using custom scene list, applying a transform

#use images/labels from these scenes only
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




#USE 3
#crop images around boxes for classification, 
#add some perturbed boxes, validate the boxes


#add more boxes to get more data
perturb_trans = transforms.AddPerturbedBoxes()#can add custom perturbations
#make sure boxes are valid 
validate_trans = transforms.ValidateMinMaxBoxes()
#compose the two transformations, validate after peturbing
target_trans = transforms.Compose([perturb_trans,validate_trans])

trainset = active_vision_dataset.AVD(root='/playpen/ammirato/Data/RohitData', 
                                    target_transform=target_trans,
                                    classification=True)

#get a list of images and their labels 
images,labels = trainset[0]

image1 = images[0]
image2 = images[1]


print('\nFirst classification image id: ' + str(labels[0]))




