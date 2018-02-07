import os
import cv2
import json

root = '/net/bvisionserver3/playpen10/ammirato/Data/HalvedRohitData/'
new_root = '/net/bvisionserver3/playpen10/ammirato/Data/HalvedRohitData/'
#new_root = '/playpen/ammirato/Data/HalvedRohitData/'





scene_list=[
#              'Home_001_1',
#              'Home_001_2',
#              'Home_003_1',
#              'Home_003_2',
#              'Home_004_1',
#              'Home_004_2',
#              'Home_005_1',
#              'Home_005_2',
#              'Home_014_1',
#              'Home_014_2',
#              'Home_008_1',
#              'Home_002_1',
#              'Home_006_1',
#              'Office_001_1',
                'Gen_004_2',
                'Gen_003_1',
                'Gen_003_2',
                'Gen_002_1',
                'Gen_002_2',
             ]




for scene in scene_list:

    image_names = os.listdir(os.path.join(root,
                                          scene,
                                          'jpg_rgb'))
    #image_names = os.listdir(os.path.join(root,
    #                                      scene,
    #                                      'high_res_depth'))

    with open(os.path.join(root,scene,'annotations.json')) as f:
        annotations = json.load(f)

    save_path = os.path.join(new_root,scene)
    img_save_path = os.path.join(new_root,scene,'jpg_rgb')
    #img_save_path = os.path.join(new_root,scene,'high_res_depth')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(img_save_path):
        os.makedirs(img_save_path)


    for il,image_name in enumerate(image_names):

        #load image, halve it, save it
        img = cv2.imread(os.path.join(root,scene, 
                                      'jpg_rgb',image_name))
        #img = cv2.imread(os.path.join(root,scene, 
        #                              'high_res_depth',image_name))


        if max(img.shape) > 1000:
           
            print '{}/{}'.format(il, len(image_names))
 
            img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
            cv2.imwrite(os.path.join(img_save_path,image_name), img)


            target = annotations[image_name]['bounding_boxes']

            for il in range(len(target)):
                box = target[il]
                box[0] = int(box[0]/2)
                box[1] = int(box[1]/2)
                box[2] = int(box[2]/2)
                box[3] = int(box[3]/2)
                target[il] = box 
            annotations[image_name]['bounding_boxes'] = target
       

    with open(os.path.join(save_path,'annotations.json'),'w') as f:
        json.dump(annotations,f)  
    
