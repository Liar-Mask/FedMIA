import numpy as np
from numpy import random
import numpy as np
import torch
from torch.utils.data import Subset


np.random.seed(1)

def wm_iid(dataset, num_users, num_back):
    """
    Sample I.I.D. client data from watermark dataset
    """
    num_items = min(num_back, int(len(dataset)/num_users))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_iid_MIA(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    all_idx0=all_idxs
    train_idxs=[]
    val_idxs=[]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        train_idxs.append(list(dict_users[i] ))
        all_idxs = list(set(all_idxs) - dict_users[i])
        val_idxs.append(list(set(all_idx0)-dict_users[i]))
    return dict_users, train_idxs, val_idxs

def cifar_beta(dataset, beta, n_clients):  
    #beta = 0.1, n_clients = 10
    label_distributions = []
    for y in range(len(dataset.dataset.classes)):
    # 根据狄利克雷分布获取n_clients* 数据集标签个数的分布矩阵
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))  
    
    labels = np.array(dataset.dataset.targets).astype(np.int32) #获取标签（数值） [19 29  0 ...  3  7 73]
    #print("labels:",labels)
    client_idx_map = {i:{} for i in range(n_clients)}
    client_size_map = {i:{} for i in range(n_clients)}
    #print("classes:",dataset.dataset.classes)
    for y in range(len(dataset.dataset.classes)): # classes: ['apple', 'aquarium_fish', 'baby', 'bear', ...]100个
    #for y in range(dataset.__len__):
        label_y_idx = np.where(labels == y)[0]
        label_y_size = len(label_y_idx) # value: 500
        #print(label_y_size)
        
        sample_size = (label_distributions[y]*label_y_size).astype(np.int32) # 根据分布矩阵为每个client划分label为y的samples个数
        #print(sample_size)
        sample_size[n_clients-1] += label_y_size - np.sum(sample_size) #最后一个client差值求出
        #print(sample_size)
        for i in range(n_clients):
            client_size_map[i][y] = sample_size[i] # 保存到client_size_map

        """在确定好sample_size之后，根据个数与数据集sample的index建立映射"""
        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size) # 第i个元素为第i个客户端拥有的samples的起始序号，以此划分samples的indexs
        for i in range(n_clients):
            client_idx_map[i][y] = label_y_idx[(sample_interval[i-1] if i>0 else 0):sample_interval[i]]

    train_idxs=[]
    val_idxs=[]    
    client_datasets = []
    all_idxs=[i for i in range(len(dataset))]
    for i in range(n_clients):
        client_i_idx = np.concatenate(list(client_idx_map[i].values()))
        np.random.shuffle(client_i_idx)
        subset = Subset(dataset.dataset, client_i_idx)
        client_datasets.append(subset)
        # save the idxs for attack
        train_idxs.append(client_i_idx)
        val_idxs.append(list(set(all_idxs)-set(client_i_idx)))

    return client_datasets, train_idxs, val_idxs




def cifar_beta_tmp(dataset, beta, n_clients):  
     #beta = 0.1, n_clients = 10
    # labels = np.array(dataset.dataset.targets).astype(np.int32) # TODO
    # TODO
    labels = np.array(dataset.tensors[1]).astype(np.int32)
    num_classes=max(labels)+1
    print(num_classes,labels.shape)
    # assert 0

    label_distributions = []
    for y in range(num_classes):# TODO
    #for y in range(dataset.__len__):
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))  
    
    
    #print(labels)
    client_idx_map = {i:{} for i in range(n_clients)}
    client_size_map = {i:{} for i in range(n_clients)}
    #print(dataset.dataset.classes)
    for y in range(num_classes): # TODO
    #for y in range(dataset.__len__):
        label_y_idx = np.where(labels == y)[0]
        label_y_size = len(label_y_idx)
        #print(label_y_size)
        
        sample_size = (label_distributions[y]*label_y_size).astype(np.int32)
        #print(sample_size)
        sample_size[n_clients-1] += len(label_y_idx) - np.sum(sample_size)
        #print(sample_size)
        for i in range(n_clients):
            client_size_map[i][y] = sample_size[i]

        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size)
        for i in range(n_clients):
            client_idx_map[i][y] = label_y_idx[(sample_interval[i-1] if i>0 else 0):sample_interval[i]]

    train_idxs=[]
    val_idxs=[]    
    client_datasets = []
    all_idxs=[i for i in range(len(dataset))]
    print(dataset.tensors[0].shape)
    for i in range(n_clients):
        client_i_idx = np.concatenate(list(client_idx_map[i].values()))
        np.random.shuffle(client_i_idx)
        subset = Subset(dataset, client_i_idx)
        # print(subset)
        tmp=subset[[1,2,3]]
        print(tmp[0].shape,tmp[1].shape)
        # assert 0
        client_datasets.append(subset)
        # save the idxs for attack
        train_idxs.append(client_i_idx)
        val_idxs.append(list(set(all_idxs)-set(client_i_idx)))

    return client_datasets, train_idxs, val_idxs

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    """ 
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

