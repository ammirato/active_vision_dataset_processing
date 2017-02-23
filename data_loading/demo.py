import torch
import torchvision
import torchvision.transforms as transforms
import active_vision_dataset


#USE 1
##########from pytorch tutorial

#create a transform to normalize the images
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])

#create the dataset instance
#use the default training scenes
trainset = active_vision_dataset.AVD(root='/playpen/ammirato/Data/RohitData', 
                                    train=True, transform=transform)

#create a torch Data loader instance
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True)

#create a torch iterator over the data
dataiter = iter(trainloader)

#get one batch of images and labels
images,labels = dataiter.next()






#USE 2 
##########not using pytorch, using custom scene list

#use images/labels from these scenes only
scene_list = ['Home_01_1', 'Home_02_1', 'Home_03_1']

trainset = active_vision_dataset.AVD(root='/playpen/ammirato/Data/RohitData', 
                                    scene_list=scene_list)

#get an image and its label(s) 
image,labels = trainset[0]
#get the next image and its label(s)
image2,labels2 = trainset[1]

