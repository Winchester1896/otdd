import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, transforms

# def create_dataset_format(dataset_folder):
 
def listdirs(rootdir):
    img_list = []
    for i in ['testing', 'training', 'validation']:
        for j in ['0', '1', '2', '3', '4', '5']:
            temp_dir = rootdir + '/' + i + '/' + j + '/'
            temp_list = os.listdir(temp_dir)
            for img in temp_list:
                if img[-3:] == 'JPG' or img[-3:] == 'jpg':
                    img_list.append(temp_dir + img)
    return img_list
 

def write_csv_file(filename, data):
    with open(filename, 'w') as f:
        fw = csv.writer(f)
        fw.writerows(data)
    f.close()

 
def make_new_dataset(rootdir, newdir):
    if not os.path.exists(newdir):
        os.mkdir(newdir)
    csv_data = []
    img_count = 1000000
    img_list = listdirs(rootdir)
    for ele in img_list:
        temp = ele.split('/')
        label = temp[-2]
        new_name = newdir + '/img_' + str(img_count)[1:] + '.jpg'
        shutil.move(ele, new_name)
        img_count += 1
        img_data = [new_name, label]
        # print(img_data)
        csv_data.append(img_data)
    print(len(csv_data))
    write_csv_file('soybean_leaf_diseases_dataset.csv', csv_data)


def to_pytorch_dataset_format():
    # Write transform for image
    data_transform = transforms.Compose([
        # Resize the images to 64x64
        transforms.Resize(size=(64, 64)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])

    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                    transform=data_transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                    transform=data_transform)

    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
    

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label