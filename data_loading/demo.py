import torch
import torchvision
import torchvision.transforms as transforms
import active_vision_dataset



#from pytorch tutorial


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

