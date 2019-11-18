from torch.utils.data import Dataset
import os
import csv
import numpy as np
from skimage import io
from skimage.transform import resize
class input_data(Dataset):

    def __init__(self, root_dir,type, image_height = 120, image_width = 224, csv_file = "data" ):
        
        self.root_dir = root_dir
        self.type = type
        self.csv_file = csv_file + "_" +self.type + ".csv"
        self.image_height = image_height
        self.image_width = image_width
        self.number_of_class = len(os.listdir(self.root_dir))
        print("total number of classes in "+self.type+" are : ", self.number_of_class )
        self.make_csv()
        with open(self.csv_file,'r') as dest_f:
            data_iter = csv.reader(dest_f)
            self.data = [data for data in data_iter]
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx][0]
        image = io.imread(img_name)
        image = resize(image, (self.image_height, self.image_width))
        image = np.moveaxis(image, -1, 0)
        landmarks = np.array(int(self.data[idx][1]))
        # sample = {'image': image, 'landmarks': landmarks}

        return image, landmarks, img_name, self.number_of_class
    
    def make_csv(self):
        directory = os.getcwd()
        os.chdir(self.root_dir)
        classes = os.listdir()
        csv_file = []
        for i,class_ in enumerate(classes):
            path = self.root_dir + "/" + class_
            for j,data in enumerate(os.listdir(path)):
                img = [path+"/"+data,str(i)]
                csv_file.append(img)
        print("size of data", len(csv_file))
        csv_file = np.array(csv_file)
        np.random.shuffle(csv_file)
        os.chdir(directory)
        with open(self.csv_file, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(csv_file)
        writeFile.close()

