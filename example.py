from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.distance import DatasetDistance
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
import torch.utils.data as data_utils
import time
import os
import csv
import shutil
import torch
import numpy as np
from geomloss import SamplesLoss
import random
import math
import argparse
import subprocess




#random.seed(14)
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", type=int, default=0) # 0 for print similarity table, 1 for add new dataset then print table
ap.add_argument("-d", "--dataset", type=str, default=None) # new dataset info to be added. dataset_name, url
ap.add_argument("-u", "--unitsize", type=int, default=2000)
#ap.add_argument("-o", "--outpath", type=str)
args = vars(ap.parse_args())


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


# rootdir = 'soybean_leaf_diseases_dataset'
# make_new_dataset(rootdir, 'soybean_leaf_diseases', img_list)

def load_npz_dataset(ds_path):
    dataset = np.load(ds_path)
    x_train, y_train, x_test, y_test = dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test']
    x_data = np.concatenate((x_train, x_test))
    y_data = np.concatenate((y_train, y_test))
    return x_data, y_data


def custom_dataset_loader(ds_path, re_size):
    '''
    loads image data set from image folder. min_size=1000. max_size=2000. per unit
    get dataset size first -> select split dataset strategy -> return dataloader list
    '''
    BATCHSIZE = 64
    UNITSIZE = args["unitsize"]
    data_transform = transforms.Compose([
                                transforms.Resize(size=(re_size, re_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                              ])
    if ds_path[-1] != '/':
        ds_path += '/'
    train_set = datasets.ImageFolder(root=ds_path + 'train',transform=data_transform)
    test_set = datasets.ImageFolder(root=ds_path + 'test',transform=data_transform)
    entire_set = train_set
    dataset_loader = []
    if len(entire_set) < UNITSIZE:
        dataset_loader.append(DataLoader(entire_set, shuffle=True, batch_size = BATCHSIZE))
    else:
        unit_size = UNITSIZE
        indices = []
        for i in range(len(entire_set)):
            indices.append(int(i))
        random.shuffle(indices)
        for i in range(len(entire_set)//unit_size):
            tmp_indices = indices[i*unit_size:(i+1)*unit_size]
            tmp_indices = torch.tensor(tmp_indices)
            unit_set = data_utils.Subset(entire_set, tmp_indices)
            unit_loader = DataLoader(unit_set, shuffle=True, batch_size=BATCHSIZE)
            dataset_loader.append(unit_loader)
    entire_set = test_set
    if len(entire_set) < UNITSIZE:
        dataset_loader.append(DataLoader(entire_set, shuffle=True, batch_size = BATCHSIZE))
    else:
        unit_size = UNITSIZE
        indices = []
        for i in range(len(entire_set)):
            indices.append(int(i))
        random.shuffle(indices)
        for i in range(len(entire_set)//unit_size):
            tmp_indices = indices[i*unit_size:(i+1)*unit_size]
            tmp_indices = torch.tensor(tmp_indices)
            unit_set = data_utils.Subset(entire_set, tmp_indices)
            unit_loader = DataLoader(unit_set, shuffle=True, batch_size=BATCHSIZE)
            dataset_loader.append(unit_loader)
    return dataset_loader


def dataset_from_numpy(X, Y, num_sample, classes = None):
    data_transform = transforms.Compose([
                                transforms.Resize(size=(28, 28)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                              ])
    X, Y = X[:num_sample], Y[:num_sample]
    targets =  torch.LongTensor(list(Y))
    # ds = TensorDataset(torch.from_numpy(X).type(torch.FloatTensor),targets)
    ds = Dataset(X, Y, transform=data_transform)
    ds.targets =  targets
    ds.classes = classes if classes is not None else [i for i in range(len(np.unique(Y)))]
    return ds


def get_otdd(src_path, tgt_path, img_resize):
    '''
    Load data
    FashionMNIST, USPS, MNIST, CIFAR10, 
    '''
    t0 = time.time()
    print(src_path, tgt_path)
    src_loader = custom_dataset_loader(src_path, img_resize) #soybean leaf diseases dataset, 224x224x3, 6636
    tgt_loader = custom_dataset_loader(tgt_path, img_resize) # soybean leaf defoliation dataset, 108x108x3, 97395
    # osc path : '../../data/dataset', local path : '../data/soybean_defo_dataset/dataset'
    # loaders_src  = load_torchvision_data('CIFAR10', valid_size=0, resize = 28, maxsize=2000)[0]
    # loaders_tgt  = load_torchvision_data('CIFAR10',  valid_size=0, resize = 28, maxsize=2000)[0]
    if len(src_loader) <= 3:
        src_ratio = len(src_loader)
    elif len(src_loader) <= 20:
        src_ratio = 4
    elif len(src_loader) <= 40:
        src_ratio = 5
    else:
        src_ratio = 6
    if len(tgt_loader) <= 3:
        tgt_ratio = len(tgt_loader)
    elif len(tgt_loader) <= 20:
        tgt_ratio = 4
    elif len(tgt_loader) <= 40:
        tgt_ratio = 5
    else:
        tgt_ratio = 6
    # print(len(src_loader), len(tgt_loader), src_ratio, tgt_ratio)
    src_ratio, tgt_ratio = 3, 1
    print(len(src_loader), len(tgt_loader), src_ratio, tgt_ratio)
    if src_path != tgt_path:
        
        random.shuffle(src_loader)
        random.shuffle(tgt_loader)
        total_d = 0
        count = 0
        for i in range(src_ratio):
            for j in range(tgt_ratio):
                dist = DatasetDistance(src_loader[i], tgt_loader[j],
                                    inner_ot_method = 'exact',
                                    debiased_loss = True,
                                    p = 2, entreg = 1e-1,
                                    # coupling_method='pot',
                                    device=device)

                d = dist.distance()
                #print(d)
                total_d += d
                count += 1
        avg_d = total_d / count
    else:
        random.shuffle(src_loader)
        total_d = 0
        count = 0
        for i in range(src_ratio):
            for j in range(tgt_ratio):
                if i != j:
                    dist = DatasetDistance(src_loader[i], src_loader[j],
                                        inner_ot_method = 'exact',
                                        debiased_loss = True,
                                        p = 2, entreg = 1e-1,
                                        # coupling_method='pot',
                                        device=device)

                    d = dist.distance()
                    #print(d)
                    total_d += d
                    count += 1
        avg_d = total_d / count
    t1 = time.time() - t0
    with open('otdd_results.txt', 'a+') as f:
        f.write(f'{src_path}, {tgt_path}, IMG_SIZE={img_resize}, {src_ratio}x{tgt_ratio}, OTDD={avg_d:.2f}, time={t1:.3f}s\n')
    f.close()
    print(f'{src_path}, {tgt_path}, IMG_SIZE={img_resize}, {src_ratio}x{tgt_ratio}, OTDD={avg_d:.2f}, time={t1:.3f}s')


def get_ds_name(x):
    ds = x.split('/')
    return ds[-2]


def get_similarity_table(txt_file):
    # just return a numpy array and a dataset list
    f = open(txt_file, 'r')
    lines = f.readlines()
    dataset_list = []
    lists = []
    for line in lines:
        ele = line.split(',')
        ds0 = get_ds_name(ele[0])
        ds1 = get_ds_name(ele[1])
        otdd = ele[4][6:]
        if ds0 not in dataset_list:
            dataset_list.append(ds0)
        if ds1 not in dataset_list:
            dataset_list.append(ds1)
        lists.append([ds0, ds1, otdd])
        if ds0 != ds1:
            lists.append([ds1, ds0, otdd])
    lists.sort()
    dataset_list.sort()
    otdd_array = np.zeros((len(dataset_list), len(dataset_list)))
    for l in lists:
        ds0_index, ds1_index = dataset_list.index(l[0]), dataset_list.index(l[1])
        otdd_array[ds0_index][ds1_index] = float(l[2])
    print(otdd_array)
    print(dataset_list)
    return otdd_array, dataset_list


def get_dataset_list(txt_file):
    f = open(txt_file, 'r')
    lines = f.readlines()
    dataset_list = []
    for line in lines:
        ele = line.split(',')
        ds0 = get_ds_name(ele[0])
        ds1 = get_ds_name(ele[1])
        if ds0 not in dataset_list:
            dataset_list.append(ds0)
        if ds1 not in dataset_list:
            dataset_list.append(ds1)
    return dataset_list


def get_new_dataset_info(input_str):
    tmp = input_str.split(',')
    return tmp[0], tmp[1]


def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


def download_dataset(ds_name, ds_url):
    print('downloading new dataset')
    cmd = 'wget -O ' + '../data/' + ds_name + '.tar.gz ' + ds_url
    runcmd(cmd, verbose=True)
    #cmd = 'tar -xf ' + '../data/' + ds_name + '.tar.gz'
    print('extrating dataset')
    cmd = 'python -m tarfile -e '+'../data/'+ds_name+'.tar.gz ../data/' 
    runcmd(cmd, verbose=True)


def compute_new_dataset_similarity(dataset_list, new_dataset):
    dataset_list.remove('imagenet_1k')
    if new_dataset[-1] != '/':
        new_dataset += '/'
    for i in range(len(dataset_list)):
        if dataset_list[i][-1] != '/':
            dataset_list[i] += '/'
        src_path, tgt_path = '../data/'+new_dataset, '../data/'+dataset_list[i]
        get_otdd(src_path, tgt_path, 28)
    get_otdd(src_path, src_path, 28)



if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # result_file = 'otdd_results.txt'
    # if args["mode"] == 1:
    #     new_dataset_name, new_dataset_url = get_new_dataset_info(args['dataset'])
    #     if not os.path.exists('../data/'+new_dataset_name):
    #         download_dataset(new_dataset_name, new_dataset_url)
    #     ds_list = get_dataset_list(result_file)
    #     compute_new_dataset_similarity(ds_list, new_dataset_name)
    # similarity_table, dataset_list = get_similarity_table(result_file)
    download_dataset('imagenette2', 'https://datacommons.tdai.osu.edu/api/access/datafile/5353')
    src_path, tgt_path = '../data/imagenet_1k', '../data/imagenette2'
    get_otdd(src_path, tgt_path, 28)




















