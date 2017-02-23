import torch
import torchvision
import torchvision.transforms as transforms
import active_vision_dataset
import active_vision_dataset_pytorch






#USE 1
########## non pytorch
trainset = active_vision_dataset.AVD(root='/playpen/ammirato/Data/RohitData')
                                    
#get an image and its label(s) 
image,labels = trainset[0]

print('Number of boxes: ' + str(len(labels)))
print('First box: ' + str(labels[0][0:4]))
print('First box instance id: ' + str(labels[0][4]))
print('First box difficulty: ' + str(labels[0][5]))





#USE 2 
########## not using pytorch, using custom scene list

#use images/labels from these scenes only
scene_list = ['Home_001_1', 'Home_002_1', 'Home_003_1']

trainset = active_vision_dataset.AVD(root='/playpen/ammirato/Data/RohitData', 
                                    scene_list=scene_list)

#get an image and its label(s) 
image,labels = trainset[0]
#get the next image and its label(s)
image2,labels2 = trainset[1]




#USE 3 
##########from pytorch tutorial

#create a transform to normalize the images
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])

#create the dataset instance
#use the default training scenes
#note use of PYTORCH module
trainset = active_vision_dataset_pytorch.AVD(root='/playpen/ammirato/Data/RohitData', 
                                    train=True, transform=transform)

#create a torch Data loader instance
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True)

#create a torch iterator over the data
dataiter = iter(trainloader)

#get one batch of images and labels
#NOTE labels only have one box per image, not all boxes
#see code for explanation
images,labels = dataiter.next()






