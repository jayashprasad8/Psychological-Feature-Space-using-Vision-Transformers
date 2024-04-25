import os
import datasets
import numpy as np
import cv2
import glob



def load_images_from_folder_rgb(folder):
    images = []
    print(folder)
    # for filename in glob.glob(os.path.join(folder+'/*.jpg')):
    for filename in sorted(os.listdir(folder)):
        # print(filename)
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(cv2.resize(img,(224,224)))
    return np.array(images)

def create_image_folder_dataset(root_path):
    """creates `Dataset` from image folder structure"""
    # get class names by folders names
    _CLASS_NAMES= os.listdir(root_path)
    # defines `datasets` features`
    features=datasets.Features({
                      "img": datasets.Image(),
                      "label": datasets.features.ClassLabel(names=_CLASS_NAMES),
                  })
    # temp list holding datapoints for creation
    img_data_files=[]
    label_data_files=[]
    # load images into list for creation
    for img_class in sorted(os.listdir(root_path)):
        for img in os.listdir(os.path.join(root_path,img_class)):
            path_=os.path.join(root_path,img_class,img)
            img_data_files.append(path_)
            label_data_files.append(img_class)
    # create dataset
    ds = datasets.Dataset.from_dict({"img":img_data_files,"label":label_data_files},features=features)
    return ds