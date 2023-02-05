# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import os
import scipy.io
import yaml
import torch
from torchvision import datasets, transforms
from model import ft_net, DisentangleNet
from model import load_network, load_whole_network
import numpy as np
from datasets import ChannelTripletFolder, ChannelDatasetAllDomain
from rerank_for_cluster import re_ranking
from scipy.io import loadmat
from scipy.io import savemat

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_one_feature_original(model, dataloaders, domain_order=0):
    features = torch.FloatTensor()
    for data in dataloaders:
        img_all = data
        img = img_all[:, domain_order]
        n, c, h, w = img.size()
        ff_d = torch.FloatTensor(n, 512).zero_().cuda()
        ff_s = torch.FloatTensor(n, 512).zero_().cuda()
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = img.cuda()
            ret = model(input_img)
            did_outputs = ret[1]
            sid_outputs = ret[5]
            ff_d = ff_d + did_outputs
            ff_s = ff_s + sid_outputs
        # norm feature
        fnorm = torch.norm(ff_d, p=2, dim=1, keepdim=True)
        ff_d = ff_d.div(fnorm.expand_as(ff_d))
        ff_d = ff_d.detach().cpu().float()
        fnorm = torch.norm(ff_s, p=2, dim=1, keepdim=True)
        ff_s = ff_s.div(fnorm.expand_as(ff_s))
        ff_s = ff_s.detach().cpu().float()
        features = torch.cat((features, torch.cat((ff_d, ff_s), 1)), 0)
    return features

def extract_one_feature(model, dataloaders, domain_order=0, flag='did'):
    features = torch.FloatTensor()
    if 'all' in flag:
        for data in dataloaders:
            img_all = data
            img = img_all[:, domain_order]
            n, c, h, w = img.size()
            ff_d = torch.FloatTensor(n, 512).zero_().cuda()
            ff_s = torch.FloatTensor(n, 512).zero_().cuda()
            for i in range(2):
                if (i == 1):
                    img = fliplr(img)
                input_img = img.cuda()
                did_outputs = model(input_img)[1]
                sid_outputs = model(input_img)[3]
                ff_d = ff_d + did_outputs
                ff_s = ff_s + sid_outputs
            # norm feature
            fnorm = torch.norm(ff_d, p=2, dim=1, keepdim=True)
            ff_d = ff_d.div(fnorm.expand_as(ff_d))
            ff_d = ff_d.detach().cpu().float()
            fnorm = torch.norm(ff_s, p=2, dim=1, keepdim=True)
            ff_s = ff_s.div(fnorm.expand_as(ff_s))
            ff_s = ff_s.detach().cpu().float()
            features = torch.cat((features, torch.cat((ff_d, ff_s), 1)), 0)
    else:
        if 'did' in flag:
            index = 1
        elif 'sid' in flag:
            index = 3
        else:
            print('flag = %s error!!!!!!!' % flag)
            exit()
        for data in dataloaders:
            img_all = data
            img = img_all[:, domain_order]
            n, c, h, w = img.size()
            ff = torch.FloatTensor(n, 512).zero_().cuda()
            for i in range(2):
                if (i == 1):
                    img = fliplr(img)
                input_img = img.cuda()
                outputs = model(input_img)[index]
                ff = ff + outputs
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.detach().cpu().float()
            features = torch.cat((features, ff), 0)
    return features

def extract_all_features(model, dataloaders, domain_num):
    features = torch.FloatTensor()
    cnt = 0
    for data in dataloaders:
        img_all = data
        n, c, h, w = img_all[0].size()
        cnt += n
        # print(cnt)
        ff = torch.FloatTensor(n, 512 * domain_num).zero_().cuda()
        for d in range(domain_num):
            img = img_all[d]
            f = torch.FloatTensor(n, 512).zero_().cuda()
            for i in range(2):
                if i == 1:
                    img = fliplr(img)
                input_img = img.cuda()
                outputs = model(input_img)[1]
                f = f + outputs
                # norm feature
                fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
                f = f.div(fnorm.expand_as(f))
            ff = torch.cat((ff, f), 1)
        ff = ff.detach().cpu().float()
        features = torch.cat((features, ff), 0)
    return features


def get_id(img_path, test_dir):
    camera_id = []
    labels = []
    names = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        if 'msmt' in test_dir:
            camera = filename[9:11]
        else:
            camera = filename.split('c')[1][0]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera))
        names.append(filename)
    return camera_id, labels, names


def get_features(flag='did', multi_domain=False, order=0, data_dir=None, net_loss_model=None, domain_num=None, which_epoch=None):
    print('semantic: %s get features' % flag)
    ######################################################################
    # Load Data
    # --------------------------------------------------------------------
    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_list = ['train_all_new']
    image_datasets = {
    x: ChannelTripletFolder(os.path.join(data_dir, x), data_transforms, domain_num=6, train=False) for
    x in dataset_list}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                  shuffle=False, num_workers=0) for x in dataset_list}
    use_gpu = torch.cuda.is_available()
    print('test_dir = %s   net_loss_model = %d   domain_num = %d' % (data_dir, net_loss_model, domain_num))
    dataset_path = []
    for i in range(len(dataset_list)):
        dataset_path.append(image_datasets[dataset_list[i]].imgs)
    dataset_cam = []
    dataset_label = []
    dataset_name = []
    for i in range(len(dataset_list)):
        cam, label, n = get_id(dataset_path[i], data_dir)
        dataset_cam.append(cam)
        dataset_label.append(label)
        dataset_name.append(n)

    ######################################################################
    # Load Collected data Trained model
    print('---------test-----------')
    class_num = len(os.listdir(os.path.join(data_dir, 'train_all_new')))
    sid_num = class_num
    did_num = class_num * domain_num
    did_embedding_net = ft_net(id_num=did_num)
    sid_embedding_net = ft_net(id_num=sid_num)
    model = DisentangleNet(did_embedding_net, sid_embedding_net)
    if use_gpu:
        model.cuda()

    name = ''
    if order == 0:
        model = load_whole_network(model, name, 'pretrain')
    else:
        if 'best' in which_epoch or 'last' in which_epoch:
            model = load_whole_network(model, name, which_epoch + '_' + str(net_loss_model))
        else:
            model = load_whole_network(model, name, which_epoch)
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    if multi_domain:
        for d in np.arange(domain_num):
            # Extract feature
            dataset_feature = []
            with torch.no_grad():
                for i in range(len(dataset_list)):
                    dataset_feature.append(extract_one_feature(model, dataloaders[dataset_list[i]], d, flag))

            result = {'train_f': dataset_feature[0].numpy(), 'train_label': dataset_label[0],
                      'train_cam': dataset_cam[0],
                      'train_name': dataset_name[0]}
            scipy.io.savemat(str(d) + '_' + flag + '_' + data_dir.split('/')[1] + '_pytorch_target_result.mat', result)
    else:
        with torch.no_grad():
            # Extract feature
            dataset_feature = []
            if 'all' in flag:
                dataset_feature.append(extract_one_feature(model, dataloaders[dataset_list[0]], 0, flag))
                dataset_feature.append(dataset_feature[0][:, :512])
                dataset_feature.append(dataset_feature[0][:, 512:])
                result = {'train_f': dataset_feature[0].numpy(), 'train_label': dataset_label[0],
                          'train_cam': dataset_cam[0],
                          'train_name': dataset_name[0]}
                scipy.io.savemat(str(0) + '_' + 'all' + '_' + data_dir.split('/')[1] + '_pytorch_target_result.mat',
                                 result)
                result = {'train_f': dataset_feature[1].numpy(), 'train_label': dataset_label[0],
                          'train_cam': dataset_cam[0],
                          'train_name': dataset_name[0]}
                scipy.io.savemat(str(0) + '_' + 'did' + '_' + data_dir.split('/')[1] + '_pytorch_target_result.mat',
                                 result)
                result = {'train_f': dataset_feature[2].numpy(), 'train_label': dataset_label[0],
                          'train_cam': dataset_cam[0],
                          'train_name': dataset_name[0]}
                scipy.io.savemat(str(0) + '_' + 'sid' + '_' + data_dir.split('/')[1] + '_pytorch_target_result.mat',
                                 result)
            else:
                dataset_feature.append(extract_one_feature(model, dataloaders[dataset_list[0]], 0, flag))
                result = {'train_f': dataset_feature[0].numpy(), 'train_label': dataset_label[0],
                          'train_cam': dataset_cam[0],
                          'train_name': dataset_name[0]}
                scipy.io.savemat(str(0) + '_' + flag + '_' + data_dir.split('/')[1] + '_pytorch_target_result.mat',
                                 result)

def intra_distance(features):
    x = features
    y = features
    """
    get the Euclidean Distance between to matrix
    (x-y)^2 = x^2 + y^2 - 2xy
    :param x:
    :param y:
    :return:
    """
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dist = x2 + y2 - 2 * xy
    return dist

def get_distances(src_path, tgt_path, order=-1, ratio=0.003, domain=0, flag='all'):
    print('Calculating feature distances...')
    m = loadmat(str(domain) + '_' + flag + '_' + src_path + '_pytorch_target_result.mat')
    source_features = m['train_f']
    m = loadmat(str(domain) + '_' + flag + '_' + tgt_path + '_pytorch_target_result.mat')
    target_features = m['train_f']
    rerank_dist = re_ranking(source_features, target_features, lambda_value=0.1)
    # rerank_dist = intra_distance(target_features)
    # DBSCAN cluster
    tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
    tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
    tri_mat = np.sort(tri_mat, axis=None)
    top_num = np.round(ratio * tri_mat.size).astype(int)
    eps = tri_mat[:top_num].mean()
    print('%s eps in cluster: %.3f' % (flag, eps))
    eps_list = [0]
    for i in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        print('i = %5d   %.3f' % (i, tri_mat[:int(0.001 * i * tri_mat.size)].mean()))
        eps_list.append(tri_mat[:int(0.001 * i * tri_mat.size)].mean())
    return rerank_dist, eps


if __name__ == '__main__':
    get_features(flag='all', multi_domain=False, order=0, data_dir='data/market/pytorch', net_loss_model=1, domain_num=5, which_epoch='last')
