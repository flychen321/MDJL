# -*- coding: utf-8 -*-
from __future__ import print_function, division

import math
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets.folder import default_loader


#############################################################################################################
# Channel_Dataset: It is used to get image pairs, which have the same sketch and different contents
# Parameters
#         ----------
#         domain_num: The number of augmented samples for each original one
# -----------------------------------------------------------------------------------------------------------
class Channel_Dataset(Dataset):
    def __init__(self, root, transform=None, targte_transform=None, domain_num=6):
        super(Channel_Dataset, self).__init__()
        self.image_dir = root
        self.samples = []
        self.img_label = []
        self.transform = transform
        self.targte_transform = targte_transform
        self.class_num = len(os.listdir(self.image_dir))  # the number of the class
        self.domain_num = domain_num
        print('self.class_num = %s' % self.class_num)
        dirs = os.listdir(self.image_dir)
        for dir in dirs:
            fdir = os.path.join(self.image_dir, dir)
            files = os.listdir(fdir)
            for file in files:
                self.img_label.append(int(dir))
                self.samples.append(os.path.join(self.image_dir, dir, file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = default_loader(self.samples[idx])
        label = self.img_label[idx]
        # The index_channel is used to shuffle channels of the original image
        index_channel = [[0, 1, 2],
                         [0, 2, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 1, 0]]
        p_index1 = 0.9
        if np.random.random() < p_index1:
            index1 = 0
        else:
            index1 = np.random.randint(self.domain_num)
        index2 = np.random.randint(self.domain_num)
        while index2 == index1:
            index2 = np.random.randint(self.domain_num)
        img_3channel = img.split()
        img1 = Image.merge('RGB', (img_3channel[index_channel[index1][0]], img_3channel[index_channel[index1][1]],
                                   img_3channel[index_channel[index1][2]]))
        img2 = Image.merge('RGB', (img_3channel[index_channel[index2][0]], img_3channel[index_channel[index2][1]],
                                   img_3channel[index_channel[index2][2]]))
        if self.transform is not None:
            img1 = self.transform(img1)
        if self.transform is not None:
            img2 = self.transform(img2)
        label1 = self.class_num * index1 + label
        label2 = self.class_num * index2 + label
        # The below operation can produce data with more diversity
        if np.random.randint(2) == 0:
            return img1, img2, label1, label2
        else:
            return img2, img1, label2, label1


class TripletFolder(datasets.ImageFolder):
    def __init__(self, root, transform):
        super(TripletFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets
        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)

    def _get_cam_id(self, path):
        camera_id = []
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        # camera_id = filename.split('_')[2][0:2]
        return int(camera_id) - 1

    def _get_pos_sample(self, target, index):
        pos_index = np.argwhere(self.targets == target)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = np.random.permutation(len(pos_index))
        result_path = []
        for i in range(4):
            t = i % len(rand)
            tmp_index = pos_index[rand[t]]
            result_path.append(self.samples[tmp_index][0])
        return result_path

    def _get_neg_sample(self, target):
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0, len(neg_index) - 1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        path, target = self.samples[index]
        cam = self.cams[index]
        # pos_path, neg_path
        pos_path = self._get_pos_sample(target, index)

        sample = self.loader(path)
        pos0 = self.loader(pos_path[0])
        pos1 = self.loader(pos_path[1])
        pos2 = self.loader(pos_path[2])
        pos3 = self.loader(pos_path[3])

        if self.transform is not None:
            sample = self.transform(sample)
            pos0 = self.transform(pos0)
            pos1 = self.transform(pos1)
            pos2 = self.transform(pos2)
            pos3 = self.transform(pos3)

        if self.target_transform is not None:
            target = self.target_transform(target)

        c, h, w = pos0.shape
        pos = torch.cat((pos0.view(1, c, h, w), pos1.view(1, c, h, w), pos2.view(1, c, h, w), pos3.view(1, c, h, w)), 0)
        pos_target = target
        return sample, target, pos, pos_target


class ChannelDatasetAllDomain(datasets.ImageFolder):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, root, transform, domain_num=6, train=True):
        super(ChannelDatasetAllDomain, self).__init__(root, transform)
        self.domain_num = domain_num
        self.labels = np.array(self.imgs)[:, 1]
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.class_num = len(self.classes)
        class_name = []
        for s in self.samples:
            filename = os.path.basename(s[0])
            class_name.append(filename.split('_')[0])
        self.class_name = np.asarray(class_name)

        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)
        self.transform = transform
        self.train = train
        self.root = root

    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        if 'msmt' in self.root:
            camera_id = filename[9:11]
        else:
            camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index].item()
        img = default_loader(img)
        # The index_channel is used to shuffle channels of the original image
        index_channel = [[0, 1, 2],
                         [0, 2, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 1, 0]]

        img_3channel = img.split()
        label_all_temp = []
        img_all_temp = []
        img_3channel = img.split()
        img_sub = []
        for j in range(self.domain_num):
            img = Image.merge('RGB', (img_3channel[index_channel[j][0]], img_3channel[index_channel[j][1]],
                                      img_3channel[index_channel[j][2]]))
            img_all_temp.append(img)
            label_all_temp.append(self.class_num * j + int(label))
        img_all = torch.Tensor(self.domain_num, 3, 256, 128).zero_()
        label_all = torch.Tensor(self.domain_num).long()
        if self.transform is not None:
            for i in range(self.domain_num):
                img_all[i] = self.transform(img_all_temp[i])
                label_all[i] = label_all_temp[i]

        # The below operation can produce data with more diversity
        if self.train:
            indices = np.random.permutation(self.domain_num)[:2]
            return img_all[indices], label_all[indices], indices
        else:
            return img_all

    def __len__(self):
        return len(self.imgs)


class PoseDataset(datasets.ImageFolder):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, root, transform, domain_num=6, train=True):
        super(PoseDataset, self).__init__(root, transform)
        self.domain_num = domain_num
        self.labels = np.array(self.imgs)[:, 1]
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.class_num = len(self.classes)
        class_name = []
        for s in self.samples:
            filename = os.path.basename(s[0])
            class_name.append(filename.split('_')[0])
        self.class_name = np.asarray(class_name)

        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)
        self.transform = transform
        self.train = train
        self.root = root

    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        if 'msmt' in self.root:
            camera_id = filename[9:11]
        else:
            camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index].item()
        img = default_loader(img)
        # The index_channel is used to shuffle channels of the original image
        index_channel = [[0, 1, 2],
                         [0, 2, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 1, 0]]
        # order = np.random.randint(self.domain_num)
        # img = img.split()
        # img = Image.merge('RGB', (img[index_channel[order][0]], img[index_channel[order][1]],
        #                           img[index_channel[order][2]]))
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # return img, int(label) + self.class_num * order, order

        img_3channel = img.split()
        img_all = []
        label_all = []
        for j in range(self.domain_num):
            img = Image.merge('RGB', (img_3channel[index_channel[j][0]], img_3channel[index_channel[j][1]],
                                      img_3channel[index_channel[j][2]]))
            img_all.append(img)
            label_all.append(self.class_num * j + int(label))

        if self.transform is not None:
            for i in range(self.domain_num):
                img_all[i] = self.transform(img_all[i])

        # The below operation can produce data with more diversity
        if self.train:
            indices = np.random.permutation(self.domain_num)
            return img_all[indices[0]], label_all[indices[0]], indices[0]
        else:
            return img_all

    def __len__(self):
        return len(self.imgs)


class ChannelTripletFolder(datasets.ImageFolder):
    def __init__(self, root, transform, domain_num=6, train=True):
        super(ChannelTripletFolder, self).__init__(root, transform)
        self.domain_num = domain_num
        self.labels = np.array(self.imgs)[:, 1]
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.class_num = len(self.classes)
        class_name = []
        for s in self.samples:
            filename = os.path.basename(s[0])
            class_name.append(filename.split('_')[0])
        self.class_name = np.asarray(class_name)

        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)
        self.transform = transform
        self.train = train
        self.root = root

    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        if 'msmt' in self.root:
            camera_id = filename[9:11]
        else:
            camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def _get_pos_sample(self, label, index):
        pos_index = np.argwhere(self.labels == label)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = np.random.permutation(len(pos_index))
        result_path = []
        for i in range(4):
            t = i % len(rand)
            tmp_index = pos_index[rand[t]]
            result_path.append(self.samples[tmp_index][0])
        return result_path

    def _get_neg_sample(self, label):
        neg_index = np.argwhere(self.labels != label)
        neg_index = neg_index.flatten()
        rand = random.randint(0, len(neg_index) - 1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index].item()
        cam = self.cams[index]
        # The index_channel is used to shuffle channels of the original image
        index_channel = [[0, 1, 2],
                         [0, 2, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 1, 0]]
        pos_path = self._get_pos_sample(label, index)
        img = self.loader(img)
        pos0 = self.loader(pos_path[0])
        pos1 = self.loader(pos_path[1])
        pos2 = self.loader(pos_path[2])
        pos3 = self.loader(pos_path[3])

        img_original = []
        img_original.append(img)
        img_original.append(pos0)
        img_original.append(pos1)
        img_original.append(pos2)
        img_original.append(pos3)
        img_all_temp = []
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            img_sub = []
            for j in range(self.domain_num):
                img = Image.merge('RGB', (img_3channel[index_channel[j][0]], img_3channel[index_channel[j][1]],
                                          img_3channel[index_channel[j][2]]))
                # img = Image.fromarray(np.vstack((np.array(img)[:64], np.array(img_original[i])[64:])))
                img = Image.fromarray(np.vstack((np.array(img_original[i])[:64], np.array(img)[64:])))
                img_sub.append(img)
            img_all_temp.append(img_sub)

        img_all = torch.Tensor(len(img_all_temp), self.domain_num, 3, 256, 128).zero_()
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num):
                    img_all[i][j] = self.transform(img_all_temp[i][j])

        r_img = img_all[0]
        r_pos = img_all[1:]

        if self.label_to_indices[label].shape[0] > 1:
            index_two = np.random.choice(list(set(self.label_to_indices[label]) - set([index])), 1, replace=False)
        else:
            index_two = np.random.choice(self.label_to_indices[label], 1, replace=False)
        img2, label2 = self.data[index_two[0]], self.labels[index_two[0]].item()
        img2 = default_loader(img2)
        img_original = []
        img_original.append(img2)
        label_original = []
        label_original.append(label2)
        img_all_temp = []
        label_all = []
        label_all2 = torch.Tensor(len(img_original), self.domain_num).long()
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            label = label_original[i]
            img_sub = []
            label_sub = []
            for j in range(self.domain_num):
                img = Image.merge('RGB', (img_3channel[index_channel[j][0]], img_3channel[index_channel[j][1]],
                                          img_3channel[index_channel[j][2]]))
                # img = Image.fromarray(np.vstack((np.array(img)[:64], np.array(img_original[i])[64:])))
                img = Image.fromarray(np.vstack((np.array(img_original[i])[:64], np.array(img)[64:])))
                img_sub.append(img)
                label_sub.append(self.class_num * j + int(label))
                label_all2[i][j] = self.class_num * j + int(label)
            img_all_temp.append(img_sub)
            label_all.append(label_sub)

        img_all2 = torch.Tensor(len(img_original), self.domain_num, 3, 256, 128)
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num):
                    img_all2[i][j] = self.transform(img_all_temp[i][j])

        # The below operation can produce data with more diversity
        # indices = np.random.permutation(self.domain_num)
        indices = np.arange(self.domain_num)
        if self.train:
            # return r_img[indices[:2]], r_pos[:, indices[:2]], img_all2[0, indices[:2]], \
            #        label_all2[0, indices[:2]], label_all2[0, indices[:2]], label_all2[0, indices[:2]]
            return r_img, r_pos, img_all2[0], \
                   label_all2[0], label_all2[0], label_all2[0]
        else:
            return r_img

    def __len__(self):
        return len(self.imgs)


class ChannelTripletFolder_half(datasets.ImageFolder):
    def __init__(self, root, transform, domain_num=4, train=True):
        super(ChannelTripletFolder_half, self).__init__(root, transform)
        self.domain_num = 4
        self.labels = np.array(self.imgs)[:, 1]
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.class_num = len(self.classes)
        class_name = []
        for s in self.samples:
            filename = os.path.basename(s[0])
            class_name.append(filename.split('_')[0])
        self.class_name = np.asarray(class_name)

        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)
        self.transform = transform
        self.train = train
        self.root = root

    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        if 'msmt' in self.root:
            camera_id = filename[9:11]
        else:
            camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def _get_pos_sample(self, label, index):
        pos_index = np.argwhere(self.labels == label)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = np.random.permutation(len(pos_index))
        result_path = []
        for i in range(4):
            t = i % len(rand)
            tmp_index = pos_index[rand[t]]
            result_path.append(self.samples[tmp_index][0])
        return result_path

    def _get_neg_sample(self, label):
        neg_index = np.argwhere(self.labels != label)
        neg_index = neg_index.flatten()
        rand = random.randint(0, len(neg_index) - 1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index].item()
        cam = self.cams[index]
        # The index_channel is used to shuffle channels of the original image
        index_channel = [[0, 1, 2],
                         [0, 2, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 1, 0]]
        pos_path = self._get_pos_sample(label, index)
        img = self.loader(img)
        pos0 = self.loader(pos_path[0])
        pos1 = self.loader(pos_path[1])
        pos2 = self.loader(pos_path[2])
        pos3 = self.loader(pos_path[3])

        img_original = []
        img_original.append(img)
        img_original.append(pos0)
        img_original.append(pos1)
        img_original.append(pos2)
        img_original.append(pos3)
        img_all_temp = []
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            img_sub = []
            for j in range(self.domain_num):
                if j == 0:
                    img = img_original[i]
                    img_sub.append(img)
                elif j == 1:
                    c = Image.fromarray((np.array(img_3channel[0]) * 0.5).astype(np.uint8))
                    img = Image.merge('RGB', (c, img_3channel[1], img_3channel[2]))
                    img_sub.append(img)
                elif j == 2:
                    c = Image.fromarray((np.array(img_3channel[1]) * 0.5).astype(np.uint8))
                    img = Image.merge('RGB', (img_3channel[0], c, img_3channel[2]))
                    img_sub.append(img)
                elif j == 3:
                    c = Image.fromarray((np.array(img_3channel[2]) * 0.5).astype(np.uint8))
                    img = Image.merge('RGB', (img_3channel[0], img_3channel[1], c))
                    img_sub.append(img)
            img_all_temp.append(img_sub)

        img_all = torch.Tensor(len(img_all_temp), self.domain_num, 3, 256, 128).zero_()
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num):
                    img_all[i][j] = self.transform(img_all_temp[i][j])

        r_img = img_all[0]
        r_pos = img_all[1:]

        if self.label_to_indices[label].shape[0] > 1:
            index_two = np.random.choice(list(set(self.label_to_indices[label]) - set([index])), 1, replace=False)
        else:
            index_two = np.random.choice(self.label_to_indices[label], 1, replace=False)
        img2, label2 = self.data[index_two[0]], self.labels[index_two[0]].item()
        img2 = default_loader(img2)
        img_original = []
        img_original.append(img2)
        label_original = []
        label_original.append(label2)
        img_all_temp = []
        label_all = []
        label_all2 = torch.Tensor(len(img_original), self.domain_num).long()
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            label = label_original[i]
            img_sub = []
            label_sub = []
            for j in range(self.domain_num):
                if j == 0:
                    img = img_original[i]
                    img_sub.append(img)
                elif j == 1:
                    c = Image.fromarray((np.array(img_3channel[0]) * 0.5).astype(np.uint8))
                    img = Image.merge('RGB', (c, img_3channel[1], img_3channel[2]))
                    img_sub.append(img)
                elif j == 2:
                    c = Image.fromarray((np.array(img_3channel[1]) * 0.5).astype(np.uint8))
                    img = Image.merge('RGB', (img_3channel[0], c, img_3channel[2]))
                    img_sub.append(img)
                elif j == 3:
                    c = Image.fromarray((np.array(img_3channel[2]) * 0.5).astype(np.uint8))
                    img = Image.merge('RGB', (img_3channel[0], img_3channel[1], c))
                    img_sub.append(img)
                img_sub.append(img)
                label_sub.append(self.class_num * j + int(label))
                label_all2[i][j] = self.class_num * j + int(label)
            img_all_temp.append(img_sub)
            label_all.append(label_sub)

        img_all2 = torch.Tensor(len(img_original), self.domain_num, 3, 256, 128)
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num):
                    img_all2[i][j] = self.transform(img_all_temp[i][j])

        # The below operation can produce data with more diversity
        # indices = np.random.permutation(self.domain_num)
        indices = np.arange(self.domain_num)
        if self.train:
            # return r_img[indices[:2]], r_pos[:, indices[:2]], img_all2[0, indices[:2]], \
            #        label_all2[0, indices[:2]], label_all2[0, indices[:2]], label_all2[0, indices[:2]]
            return r_img, r_pos, img_all2[0], \
                   label_all2[0], label_all2[0], label_all2[0]
        else:
            return r_img

    def __len__(self):
        return len(self.imgs)


class ChannelTripletFolder_mean(datasets.ImageFolder):
    def __init__(self, root, transform, domain_num=4, train=True):
        super(ChannelTripletFolder_mean, self).__init__(root, transform)
        self.domain_num = 4
        self.labels = np.array(self.imgs)[:, 1]
        self.data = np.array(self.imgs)[:, 0]
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.class_num = len(self.classes)
        class_name = []
        for s in self.samples:
            filename = os.path.basename(s[0])
            class_name.append(filename.split('_')[0])
        self.class_name = np.asarray(class_name)

        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)
        self.transform = transform
        self.train = train
        self.root = root

    def _get_cam_id(self, path):
        filename = os.path.basename(path)
        if 'msmt' in self.root:
            camera_id = filename[9:11]
        else:
            camera_id = filename.split('c')[1][0]
        return int(camera_id) - 1

    def _get_pos_sample(self, label, index):
        pos_index = np.argwhere(self.labels == label)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = np.random.permutation(len(pos_index))
        result_path = []
        for i in range(4):
            t = i % len(rand)
            tmp_index = pos_index[rand[t]]
            result_path.append(self.samples[tmp_index][0])
        return result_path

    def _get_neg_sample(self, label):
        neg_index = np.argwhere(self.labels != label)
        neg_index = neg_index.flatten()
        rand = random.randint(0, len(neg_index) - 1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index].item()
        cam = self.cams[index]
        # The index_channel is used to shuffle channels of the original image
        index_channel = [[0, 1, 2],
                         [0, 2, 1],
                         [1, 0, 2],
                         [1, 2, 0],
                         [2, 0, 1],
                         [2, 1, 0]]
        pos_path = self._get_pos_sample(label, index)
        img = self.loader(img)
        pos0 = self.loader(pos_path[0])
        pos1 = self.loader(pos_path[1])
        pos2 = self.loader(pos_path[2])
        pos3 = self.loader(pos_path[3])

        img_original = []
        img_original.append(img)
        img_original.append(pos0)
        img_original.append(pos1)
        img_original.append(pos2)
        img_original.append(pos3)
        img_all_temp = []
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            img_sub = []
            for j in range(self.domain_num):
                if j == 0:
                    img = img_original[i]
                    img_sub.append(img)
                elif j == 1:
                    c = Image.fromarray(
                        ((np.array(img_3channel[1]) * 0.5) + (np.array(img_3channel[2]) * 0.5)).astype(np.uint8))
                    img = Image.merge('RGB', (c, img_3channel[1], img_3channel[2]))
                    img_sub.append(img)
                elif j == 2:
                    c = Image.fromarray(
                        ((np.array(img_3channel[0]) * 0.5) + (np.array(img_3channel[2]) * 0.5)).astype(np.uint8))
                    img = Image.merge('RGB', (img_3channel[0], c, img_3channel[2]))
                    img_sub.append(img)
                elif j == 3:
                    c = Image.fromarray(
                        ((np.array(img_3channel[0]) * 0.5) + (np.array(img_3channel[1]) * 0.5)).astype(np.uint8))
                    img = Image.merge('RGB', (img_3channel[0], img_3channel[1], c))
                    img_sub.append(img)
            img_all_temp.append(img_sub)

        img_all = torch.Tensor(len(img_all_temp), self.domain_num, 3, 256, 128).zero_()
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num):
                    img_all[i][j] = self.transform(img_all_temp[i][j])

        r_img = img_all[0]
        r_pos = img_all[1:]

        if self.label_to_indices[label].shape[0] > 1:
            index_two = np.random.choice(list(set(self.label_to_indices[label]) - set([index])), 1, replace=False)
        else:
            index_two = np.random.choice(self.label_to_indices[label], 1, replace=False)
        img2, label2 = self.data[index_two[0]], self.labels[index_two[0]].item()
        img2 = default_loader(img2)
        img_original = []
        img_original.append(img2)
        label_original = []
        label_original.append(label2)
        img_all_temp = []
        label_all = []
        label_all2 = torch.Tensor(len(img_original), self.domain_num).long()
        for i in range(len(img_original)):
            img_3channel = img_original[i].split()
            label = label_original[i]
            img_sub = []
            label_sub = []
            for j in range(self.domain_num):
                if j == 0:
                    img = img_original[i]
                    img_sub.append(img)
                elif j == 1:
                    c = Image.fromarray(
                        ((np.array(img_3channel[1]) * 0.5) + (np.array(img_3channel[2]) * 0.5)).astype(np.uint8))
                    img = Image.merge('RGB', (c, img_3channel[1], img_3channel[2]))
                    img_sub.append(img)
                elif j == 2:
                    c = Image.fromarray(
                        ((np.array(img_3channel[0]) * 0.5) + (np.array(img_3channel[2]) * 0.5)).astype(np.uint8))
                    img = Image.merge('RGB', (img_3channel[0], c, img_3channel[2]))
                    img_sub.append(img)
                elif j == 3:
                    c = Image.fromarray(
                        ((np.array(img_3channel[0]) * 0.5) + (np.array(img_3channel[1]) * 0.5)).astype(np.uint8))
                    img = Image.merge('RGB', (img_3channel[0], img_3channel[1], c))
                    img_sub.append(img)
                img_sub.append(img)
                label_sub.append(self.class_num * j + int(label))
                label_all2[i][j] = self.class_num * j + int(label)
            img_all_temp.append(img_sub)
            label_all.append(label_sub)

        img_all2 = torch.Tensor(len(img_original), self.domain_num, 3, 256, 128)
        if self.transform is not None:
            for i in range(len(img_all_temp)):
                for j in range(self.domain_num):
                    img_all2[i][j] = self.transform(img_all_temp[i][j])

        # The below operation can produce data with more diversity
        # indices = np.random.permutation(self.domain_num)
        indices = np.arange(self.domain_num)
        if self.train:
            # return r_img[indices[:2]], r_pos[:, indices[:2]], img_all2[0, indices[:2]], \
            #        label_all2[0, indices[:2]], label_all2[0, indices[:2]], label_all2[0, indices[:2]]
            return r_img, r_pos, img_all2[0], \
                   label_all2[0], label_all2[0], label_all2[0]
        else:
            return r_img

    def __len__(self):
        return len(self.imgs)


#############################################################################################################
# RandomErasing: Executing random erasing on input data
# Parameters
#         ----------
#         probability: The probability that the Random Erasing operation will be performed
#         sl: Minimum proportion of erased area against input image
#         sh: Maximum proportion of erased area against input image
#         r1: Minimum aspect ratio of erased area
#         mean: Erasing value
# -----------------------------------------------------------------------------------------------------------
class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
