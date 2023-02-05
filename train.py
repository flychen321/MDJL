# -*- coding: utf-8 -*-
from __future__ import print_function, division
import time
import os
import numpy as np
import yaml
from model import ft_net_dense, ft_net, DisentangleNet
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset
from model import load_network, save_network, load_whole_network, save_whole_network
from losses import SoftLabelLoss, ContrastiveLoss_diff, ContrastiveLoss_same, ContrastiveLoss_orth
from datasets import Channel_Dataset, RandomErasing, ChannelTripletFolder, ChannelDatasetAllDomain
import shutil
from finetune import generate_cluster, generate_cluster_kmeans, generate_cluster_with_semantic_clean, \
    generate_cluster_with_semantic_kmeans_clean
from get_multi_target_features import get_features, get_distances
from test import test_function
from evaluate import evaluate_function
from scipy.io import loadmat

version = torch.__version__

######################################################################
# Options
# --------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='', type=str, help='output model name')
parser.add_argument('--save_model_name', default='', type=str, help='save_model_name')
parser.add_argument('--data_stage1', default='market', type=str, help='training source dir path')
parser.add_argument('--data_stage2', default='duke', type=str, help='training target dir path')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--net_loss_model', default=1, type=int, help='net_loss_model')
parser.add_argument('--domain_num', default=4, type=int, help='domain_num, in [2,6]')
parser.add_argument('--gpu', type=str, default='0', help='GPU id to use.')
parser.add_argument('--margin', default=0.5, type=float, help='margin')
parser.add_argument('--poolsize', default=128, type=int, help='poolsize')

opt = parser.parse_args()
print('opt = %s' % opt)
print('net_loss_model = %d' % opt.net_loss_model)
print('save_model_name = %s' % opt.save_model_name)
print('domain_num = %s' % opt.domain_num)
if opt.domain_num > 6 or opt.domain_num < 2:
    print('domain_num = %s' % opt.domain_num)
    exit()
dir_name = os.path.join('./model', opt.name)
if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
######################################################################
# Load Data
# --------------------------------------------------------------------
#
transform_train_list = [
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
}

use_gpu = torch.cuda.is_available()


def get_dataset(stage=1):
    if stage == 1:
        data_path = os.path.join(data_dir_stage1, 'train_all_new')
    else:
        data_path = os.path.join(data_dir_stage2, 'train_all_cluster')
    image_datasets = {}
    image_datasets['train'] = ChannelTripletFolder(data_path,
                                                   data_transforms['train'], domain_num=opt.domain_num)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=True, num_workers=0) for x in ['train']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    return dataloaders, dataset_sizes


def get_soft_label_lsr(labels, w_main=0.7, bit_num=6):
    # w_reg is used to prevent data overflow, it a value close to zero
    w_reg = (1.0 - w_main) / bit_num
    if w_reg < 0:
        print('w_main=%s' % (w_main))
        exit()
    soft_label = np.zeros((len(labels), int(bit_num)))
    soft_label.fill(w_reg)
    for i in np.arange(len(labels)):
        soft_label[i][labels[i]] = w_main + w_reg
    return torch.Tensor(soft_label)


def train(model, criterion_identify, criterion_contrastive_same, optimizer, scheduler, sid_num, did_num, num_epochs,
          stage=1):
    global setting
    since = time.time()
    cnt = 0
    best_acc = 0.0
    best_loss = 10000.0
    best_epoch = -1
    triplet_best_acc = 0.0
    r_triplet = 0.2
    r_id = 0.6
    r_sid = 0.6
    w_main_id = 0.5
    w_main_sid = 0.5

    print('r_id = %.2f   r_sid = %.2f  r_triplet = %.2f' % (r_id, r_sid, r_triplet))
    print('w_main_id = %.2f   w_main_sid = %.2f' % (w_main_id, w_main_sid))
    print('triplet margin = %.3f' % opt.margin)

    criterion_triplet = nn.TripletMarginLoss(margin=opt.margin)
    dataloaders, dataset_sizes = get_dataset(stage)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            id_running_loss = 0.0
            id_running_corrects = 0.0
            sid_running_loss = 0.0
            sid_running_corrects = 0.0
            triplet_running_loss = 0.0
            triplet_running_corrects = 0.0
            running_margin = 0.0
            # Iterate over data.
            one_to_multi = 4
            for data in dataloaders[phase]:
                # get the inputs
                r_img, r_pos, img_all, pos_labels, r_label, label_all = data
                # indices = np.random.permutation(opt.domain_num)
                indices = np.arange(opt.domain_num)
                indices[1:] = np.random.permutation(opt.domain_num - 1) + 1
                anchor_d0 = r_img[:, indices[0]]
                pos_d0 = r_pos[:, :, indices[0]]
                same_d0 = img_all[:, indices[0]]
                anchor_d1 = r_img[:, indices[1]]
                pos_d1 = r_pos[:, :, indices[1]]
                same_d1 = img_all[:, indices[1]]

                pos_label_d0 = label_all[:, indices[0]]
                pos_label_d1 = label_all[:, indices[1]]

                now_batch_size, c, h, w = anchor_d0.shape
                if now_batch_size < opt.batchsize:  # next epoch
                    continue
                pos_d0 = pos_d0.view(4 * opt.batchsize, c, h, w)
                pos_d1 = pos_d1.view(4 * opt.batchsize, c, h, w)
                # copy pos 4times
                pos_labels = pos_labels[:, indices[0]]
                pos_labels = pos_labels.repeat(4).reshape(4, opt.batchsize)
                pos_labels = pos_labels.transpose(0, 1).reshape(4 * opt.batchsize)

                id_inputs = torch.cat(
                    (anchor_d0, anchor_d1, same_d0, same_d1),
                    0)
                id_labels = torch.cat(
                    (pos_label_d0, pos_label_d1, pos_label_d0, pos_label_d1),
                    0)
                sid_labels = id_labels % sid_num
                id_labels_soft = get_soft_label_lsr(id_labels, w_main=w_main_id, bit_num=did_num)
                sid_labels_soft = get_soft_label_lsr(sid_labels, w_main=w_main_sid, bit_num=sid_num)
                now_batch_size, c, h, w = id_inputs.shape
                if now_batch_size // one_to_multi < opt.batchsize:  # next epoch
                    print('continue')
                    continue
                if use_gpu:
                    id_inputs = id_inputs.cuda()
                    id_labels_soft = id_labels_soft.cuda()
                    sid_labels_soft = sid_labels_soft.cuda()
                    triplet_inputs0 = anchor_d0.cuda()
                    triplet_pos0 = pos_d0.cuda()
                    triplet_inputs1 = anchor_d1.cuda()
                    triplet_pos1 = pos_d1.cuda()
                    labels = pos_label_d0.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                id_outputs, id_features, sid_outputs, sid_f = model(id_inputs)
                _, id_preds = torch.max(id_outputs.detach(), 1)
                _, sid_preds = torch.max(sid_outputs.detach(), 1)
                loss_did = criterion_identify(id_outputs, id_labels_soft)
                loss_sid = criterion_identify(sid_outputs, sid_labels_soft)

                loss_id = r_id * loss_did + r_sid * loss_sid

                outputs0, f0, _, _ = model(triplet_inputs0)
                _, pf0, _, _ = model(triplet_pos0)
                # outputs1, f1, _, _ = model(triplet_inputs1)
                # _, pf1, _, _ = model(triplet_pos1)
                #
                # f = torch.cat((f0, f1), 1)
                # pf = torch.cat((pf0, pf1), 1)
                f = f0
                pf = pf0
                neg_labels = pos_labels
                now_batch_size //= one_to_multi
                # hard-neg
                # ----------------------------------
                nf_data = pf  # 128*512
                # 128 is too much, we use pool size = 64
                rand = np.random.permutation(4 * opt.batchsize)[0:opt.poolsize]
                nf_data = nf_data[rand, :]
                neg_labels = neg_labels[rand]
                nf_t = nf_data.transpose(0, 1)  # 512*128
                score = torch.mm(f.data, nf_t)  # cosine 32*128
                score, rank = score.sort(dim=1, descending=True)  # score high == hard
                labels_cpu = labels.cpu()
                nf_hard = torch.zeros(f.shape).cuda()
                for k in range(now_batch_size):
                    hard = rank[k, :]
                    for kk in hard:
                        now_label = neg_labels[kk]
                        anchor_label = labels_cpu[k]
                        if now_label != anchor_label:
                            nf_hard[k, :] = nf_data[kk, :]
                            break

                # hard-pos
                # ----------------------------------
                pf_hard = torch.zeros(f.shape).cuda()  # 32*512
                for k in range(now_batch_size):
                    pf_data = pf[4 * k:4 * k + 4, :]
                    pf_t = pf_data.transpose(0, 1)  # 512*4
                    ff = f.data[k, :].reshape(1, -1)  # 1*512
                    score = torch.mm(ff, pf_t)  # cosine
                    score, rank = score.sort(dim=1, descending=False)  # score low == hard
                    pf_hard[k, :] = pf_data[rank[0][0], :]

                # loss
                # ---------------------------------
                pscore = torch.sum(f * pf_hard, dim=1)
                nscore = torch.sum(f * nf_hard, dim=1)
                loss_triplet = criterion_triplet(f, pf_hard, nf_hard)

                loss = loss_id + r_triplet * loss_triplet
                if cnt % 1000 == 0 or torch.isnan(loss):
                    print('cnt = %5d   loss   = %.4f  loss_id = %.4f' % (
                        cnt, loss.cpu().detach().numpy(), loss_id.cpu().detach().numpy(),
                    ))
                    print('loss_did = %.4f  loss_sid = %.4f' % (
                        loss_did.cpu().detach().numpy(), loss_sid.cpu().detach().numpy()))
                    print('loss_triplet = %.4f' % (
                        loss_triplet.cpu().detach().numpy()))
                cnt += 1

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()  # * opt.batchsize
                id_running_loss += loss_id.item()
                id_running_corrects += float(torch.sum(id_preds == id_labels_soft.argmax(1).detach()))
                sid_running_loss += loss_sid.item()
                sid_running_corrects += float(torch.sum(sid_preds == sid_labels_soft.argmax(1).detach()))
                triplet_running_loss += loss_triplet.item()  # * opt.batchsize
                triplet_running_corrects += float(torch.sum(pscore > nscore + opt.margin))
                running_margin += float(torch.sum(pscore - nscore))

            datasize = dataset_sizes[phase] // opt.batchsize * opt.batchsize
            epoch_loss = running_loss / datasize
            id_epoch_loss = id_running_loss / datasize
            sid_epoch_loss = sid_running_loss / datasize
            id_epoch_acc = id_running_corrects / (datasize * one_to_multi)
            sid_epoch_acc = sid_running_corrects / (datasize * one_to_multi)
            triplet_epoch_loss = triplet_running_loss / datasize
            triplet_epoch_acc = triplet_running_corrects / datasize
            epoch_margin = running_margin / datasize
            epoch_acc = (id_epoch_acc + sid_epoch_acc + triplet_epoch_acc) / 3.0

            print(
                '{} Loss: {:.4f}  Acc: {:.4f} id_loss: {:.4f}  id_acc: {:.4f} sid_epoch_loss: {:.4f}  sid_epoch_acc: {:.4f} '.format(
                    phase, epoch_loss, epoch_acc, id_epoch_loss, id_epoch_acc, sid_epoch_loss, sid_epoch_acc))
            print('now_margin: %.4f' % opt.margin)
            print('{} triplet_epoch_loss: {:.4f} triplet_epoch_acc: {:.4f} MeanMargin: {:.4f}'.format(
                phase, triplet_epoch_loss, triplet_epoch_acc, epoch_margin))
            time_elapsed = time.time() - since
            print('Training time_elapsed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            if epoch_acc > best_acc or (np.fabs(epoch_acc - best_acc) < 1e-5 and epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                save_whole_network(model, opt.name, 'best' + '_' + str(opt.net_loss_model))

            # if epoch % 10 == 9:
            save_whole_network(model, opt.name, epoch)

    time_elapsed = time.time() - since
    print('best_epoch = %s     best_loss = %s     best_acc = %s' % (best_epoch, best_loss, best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    save_whole_network(model, opt.name, 'last' + '_' + str(opt.net_loss_model))
    return model


def initial_model(stage=1):
    if stage == 1:
        class_num = opt.class_base_stage1
    else:
        class_num = opt.class_base_stage2
    sid_num = class_num
    did_num = class_num * opt.domain_num
    print('stage:%d   sid_num = %d   did_num = %d' % (stage, sid_num, did_num))
    did_embedding_net = ft_net(id_num=did_num)
    sid_embedding_net = ft_net(id_num=sid_num)
    model = DisentangleNet(did_embedding_net, sid_embedding_net)
    if use_gpu:
        model.cuda()

    # Initialize loss functions
    criterion_identify = SoftLabelLoss()
    criterion_contrastive_same = ContrastiveLoss_same()
    classifier_id = list(map(id, model.did_embedding_net.id_classifier.parameters())) \
                    + list(map(id, model.sid_embedding_net.id_classifier.parameters()))
    classifier_fc = list(map(id, model.did_embedding_net.fc.parameters())) \
                    + list(map(id, model.sid_embedding_net.fc.parameters()))
    id_params = filter(lambda p: id(p) in classifier_id, model.parameters())
    fc_params = filter(lambda p: id(p) in classifier_fc, model.parameters())
    base_params = filter(lambda p: id(p) not in classifier_id + classifier_fc, model.parameters())

    if stage == 1:
        epoch = 22
        optimizer_ft = optim.SGD([
            {'params': id_params, 'lr': 1 * opt.lr},
            {'params': fc_params, 'lr': 1 * opt.lr},
            {'params': base_params, 'lr': 0.1 * opt.lr},
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[10, 17], gamma=0.1)
    else:
        epoch = 8
        ratio_lr = 0.05
        print('ratio_lr = %.2f' % ratio_lr)
        optimizer_ft = optim.SGD([
            {'params': id_params, 'lr': ratio_lr * 1 * opt.lr},
            {'params': fc_params, 'lr': ratio_lr * 1 * opt.lr},
            {'params': base_params, 'lr': ratio_lr * 0.1 * opt.lr},
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[8], gamma=0.1)
    print('net_loss_model = %s   epoch = %3d' % (opt.net_loss_model, epoch))
    return model, criterion_identify, criterion_contrastive_same, optimizer_ft, exp_lr_scheduler, sid_num, did_num, epoch


data_dir_stage1 = os.path.join('data', opt.data_stage1, 'pytorch')
data_dir_stage2 = os.path.join('data', opt.data_stage2, 'pytorch')
stage1_train = True
stage2_train = True
for i in np.arange(1):
    if stage1_train:
        setting = i
        print('setting = %d' % setting)
        print('train stage1 ......')
        opt.class_base_stage1 = len(os.listdir(os.path.join(data_dir_stage1, 'train_all_new')))
        print('opt.class_base_stage1 = %d' % opt.class_base_stage1)
        model, criterion_identify, criterion_contrastive_same, optimizer_ft, exp_lr_scheduler, sid_num, did_num, epoch = initial_model(
            stage=1)
        model = train(model, criterion_identify, criterion_contrastive_same, optimizer_ft, exp_lr_scheduler,
                      sid_num, did_num, epoch, stage=1)
        save_whole_network(model, opt.name, 'pretrain')
        test_function(test_dir=opt.data_stage1, net_loss_model=opt.net_loss_model, domain_num=opt.domain_num,
                      which_epoch='last')
        evaluate_function()
        test_function(test_dir=opt.data_stage2, net_loss_model=opt.net_loss_model, domain_num=opt.domain_num,
                      which_epoch='last')
        evaluate_function()

for k in np.arange(1):
    if stage2_train:
        print('train stage2 ......')
        stage2_since = time.time()
        setting = k
        print('k = %d' % k)
        print('setting = %d' % setting)
        eps0_all = 0.8
        eps0_sid = 0.8
        eps0_did = 0.8
        if 'duke' in data_dir_stage2:
            min_samples = 10
        else:
            min_samples = 8
        iter_finetune_epoch = 10
        for i in np.arange(iter_finetune_epoch):
            print('current: %d    total iter_finetune_epoch: %d' % (i, iter_finetune_epoch))
            cluster_result_path = os.path.join(data_dir_stage2, 'train_all_cluster')
            # #update features
            get_features(flag='all', multi_domain=False, order=i, data_dir=data_dir_stage1,
                         net_loss_model=opt.net_loss_model,
                         domain_num=2,
                         which_epoch='last')
            get_features(flag='all', multi_domain=False, order=i, data_dir=data_dir_stage2,
                         net_loss_model=opt.net_loss_model,
                         domain_num=2,
                         which_epoch='last')
            if i == 0:
                dist_all, eps0_all = get_distances(opt.data_stage1, opt.data_stage2, i, ratio=0.003, domain=0,
                                                   flag='all')
            else:
                dist_all, _ = get_distances(opt.data_stage1, opt.data_stage2, i, ratio=0.003, domain=0,
                                            flag='all')
            # #cluster DBSCAN
            generate_cluster(cluster_result_path, dist=dist_all, eps=eps0_all, min_samples=min_samples,
                                                 data_dir=opt.data_stage2,
                                                 flag='all')
            # generate_cluster_with_semantic_kmeans_clean(cluster_result_path, dist=dist, eps=eps, min_samples=min_samples, data_dir=opt.data_stage2,
            #                  flag='all')
            data_dir_stage2 = os.path.join('data', opt.data_stage2, 'pytorch')
            opt.class_base_stage2 = len(os.listdir(cluster_result_path))
            print('opt.class_base_stage2 = %d' % opt.class_base_stage2)
            model, criterion_identify, criterion_contrastive_same, optimizer_ft, exp_lr_scheduler, sid_num, did_num, epoch = initial_model(
                stage=2)
            if i == 0:
                model = load_whole_network(model, opt.name, 'pretrain')
            else:
                model = load_whole_network(model, opt.name, 'last' + '_' + str(opt.net_loss_model))
            model = train(model, criterion_identify, criterion_contrastive_same, optimizer_ft, exp_lr_scheduler,
                          sid_num, did_num, epoch, stage=2)
            save_whole_network(model, opt.name, 'last' + '_' + str(opt.net_loss_model) + '_' + str(i))
            test_function(test_dir=opt.data_stage2, net_loss_model=opt.net_loss_model, domain_num=opt.domain_num,
                          which_epoch='last')
            evaluate_function()
            stage2_elapse = time.time() - stage2_since
            print('Finetuning elapse in {:.0f}m {:.0f}s'.format(stage2_elapse // 60, stage2_elapse % 60))
