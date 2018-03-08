import torch
import torch.utils.data
import active_vision_dataset
import transforms
import numpy as np
import matplotlib.pyplot as plt

#USE 1
#basic use for getting images/labels for detection

#only inlcude labels for instances with id in range(5)
pick_trans = transforms.PickInstances(range(5),
                                     max_difficulty=5)


trainset = active_vision_dataset.AVD(root='/playpen/ammirato/Data/RohitData',
                                             target_transform=pick_trans)
                                    
#get an image and its label(s) 
image,labels = trainset[0]

print('Boxes: ' + str(labels[0]))
print('Image Name: ' + str(labels[1]))
print('Movements: ' + str(labels[2]))


trainloader = torch.utils.data.DataLoader(trainset,
                                      batch_size=10,
                                      shuffle=True,
                                      collate_fn=active_vision_dataset.collate)
trainiter = iter(trainloader)
for il in range(100):
    imgs,labels = trainiter.next()

    
