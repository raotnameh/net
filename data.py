from torch.utils.data import Dataset
import os, glob
import csv
import numpy as np
from skimage import io
from skimage.transform import resize

import matplotlib.pyplot as plt
import random
from scipy import ndarray
from skimage import transform
from skimage import util


def random_rotation(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-2, 2)
    return transform.rotate(image_array, random_degree)

def random_noise(image_array):
    # add random noise to the image
    return util.random_noise(image_array)

def horizontal_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

class input_data(Dataset):

    def __init__(self, root_dir,type, image_height = 224, image_width = 224, noise=True):
        
        self.root_dir = root_dir
        self.noise = noise
        self.type = type
        self.csv_file = self.type + ".csv"
        self.image_height = image_height
        self.image_width = image_width
        self.number_of_class = len(os.listdir(self.root_dir))
        print(f"number of classes in {self.type} : , {self.number_of_class}")
        if os.path.exists(self.csv_file):
            with open(self.csv_file,'r') as dest_f:
                data_iter = csv.reader(dest_f)
                self.data = [data for data in data_iter]
        else: 
            print("creating csv files")
            self.make_csv()
            with open(self.csv_file,'r') as dest_f:
                data_iter = csv.reader(dest_f)
                self.data = [data for data in data_iter]

    def classes(self):
        return self.number_of_class
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx][0]
        image = io.imread(img_name)
        if self.noise == True:
            for i in range(np.random.randint(1,10)):
                image = horizontal_flip(random_rotation(random_noise(image)))
        image = resize(image, (self.image_height, self.image_width))
        image = np.moveaxis(image, -1, 0)
        landmarks = np.array(int(self.data[idx][1]))
        # sample = {'image': image, 'landmarks': landmarks}

        return image, landmarks, img_name, self.number_of_class
    
    def make_csv(self):
        classes = glob.glob(self.root_dir+"/*")
        csv_file = []
        for i,path in enumerate(classes):
            for j,data in enumerate(os.listdir(path)):
                img = [path+"/"+data,str(i)]
                csv_file.append(img)
        csv_file = np.array(csv_file)
        np.random.shuffle(csv_file)
        with open(self.csv_file, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(csv_file)
        writeFile.close()

