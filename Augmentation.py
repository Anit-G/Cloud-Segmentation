
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

data_dir = './dataset/'

def random_augmentation(img,mask):
    
    img = image.random_rotation(img,20)
    img = image.random_shift(img,0.1,0.1)
    img = image.random_zoom(img,(0.8,0.1))

    mask = image.random_rotation(mask,20)
    mask = image.random_shift(mask,0.1,0.1)
    mask = image.random_zoom(mask,(0.8,0.1))

    img = img
    mask = mask
    return img, mask
    
def get_image_and_mask(img_id):

    my_image = data_dir + 'images/' + str(img_id) + '.png'
    my_GT = data_dir + 'gt_images/' + str(img_id) + '.png'
    img = image.load_img(my_image,
                         target_size=(600,600))
    img = image.img_to_array(img)
    mask = image.load_img(my_GT,
                          grayscale=True, target_size=(600,600))
    mask = image.img_to_array(mask)

    return img, mask

def dataset():
    images = []
    gt_maps = []
    for name in os.listdir('./dataset/images'):
        id = name.split('.')[0]
        img,mask = get_image_and_mask(id)
    
        images.append(img/255)
        gt_maps.append(mask/255)

    images = np.asarray(images)
    gt_maps = np.asarray(gt_maps)
    
    return train_test_split(images,gt_maps,test_size=100)