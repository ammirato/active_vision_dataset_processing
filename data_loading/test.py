import torch
import torch.utils.data
import active_vision_dataset_pytorch
import transforms
import numpy as np
import matplotlib.pyplot as plt

#USE 1
#basic use for getting images/labels for detection

to_tensor_trans = transforms.ToTensor()
perturb_trans = transforms.AddPerturbedBoxes(num_to_add=3,
                                                changes = [[-.45,.45],
                                                           [-.45,.45], 
                                                           [-.45,.45], 
                                                           [-.45,.45]],
                                                percents=True)
target_trans = transforms.Compose([
                                       #combine_trans,
                                       #perturb_trans,
                                       to_tensor_trans])


trainset = active_vision_dataset_pytorch.AVD(root='/playpen/ammirato/Data/RohitData',
                                             transform=to_tensor_trans,
                                             target_transform=target_trans,
                                             classification=True,
                                             by_box=True)
                                    
#get an image and its label(s) 
image,labels = trainset[0]

#print('USE 1: Number of boxes: ' + str(len(labels)))
#print('USE 1: First box: ' + str(labels[0][0:4]))
#print('USE 1: First box instance id: ' + str(labels[0][4]))
#print('USE 1: First box difficulty: ' + str(labels[0][5]))


trainloader = torch.utils.data.DataLoader(trainset,
                                      batch_size=1,
                                      shuffle=True,
                                      collate_fn=active_vision_dataset_pytorch.collate)


trainiter = iter(trainloader)


for il in range(100):

    imgs,labels = trainiter.next()

    img = imgs.squeeze(0).numpy()
    img = np.transpose(img,(1,2,0))

    b = img[:,:,0] 
    g = img[:,:,1] 
    r = img[:,:,2] 

    img =  np.stack((r,g,b),axis=2)

    plt.imshow(img)
    plt.draw()
    plt.pause(.001)

    raw_input('Enter')
    
