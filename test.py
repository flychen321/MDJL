# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import os
import scipy.io
import yaml
import torch
from torchvision import datasets, transforms
from model import ft_net_dense, ft_net, DisentangleNet
from model import load_network, load_whole_network

######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='duke', type=str, help='./test_data')
parser.add_argument('--name', default='', type=str, help='save model path')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--net_loss_model', default=1, type=int, help='net_loss_model')
parser.add_argument('--domain_num', default=5, type=int, help='domain_num')
parser.add_argument('--gpu', type=str, default='0', help='GPU id to use.')
opt = parser.parse_args()
print('opt = %s' % opt)


def test_function(test_dir=None, net_loss_model=None, domain_num=None, which_epoch=None):
    if test_dir != None:
        opt.test_dir = test_dir
    if net_loss_model != None:
        opt.net_loss_model = net_loss_model
    if domain_num != None:
        opt.domain_num = domain_num
    if which_epoch != None:
        opt.which_epoch = which_epoch

    print('opt.which_epoch = %s' % opt.which_epoch)
    print('opt.test_dir = %s' % opt.test_dir)
    print('opt.name = %s' % opt.name)
    print('opt.batchsize = %s' % opt.batchsize)
    name = opt.name
    data_dir = os.path.join('data', opt.test_dir, 'pytorch')
    print('data_dir = %s' % data_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    ######################################################################
    # Load Data
    # ---------
    data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset_list = ['gallery', 'query']
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in dataset_list}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                  shuffle=False, num_workers=0) for x in dataset_list}
    class_names = image_datasets[dataset_list[1]].classes
    use_gpu = torch.cuda.is_available()

    ######################################################################
    # Extract feature
    # ----------------------
    #
    # Extract feature from  a trained model.
    #
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def extract_feature_original(model, dataloaders):
        features = torch.FloatTensor()
        for data in dataloaders:
            img, label = data
            n, c, h, w = img.size()
            ff_d = torch.FloatTensor(n, 512).zero_().cuda()
            for i in range(2):
                if (i == 1):
                    img = fliplr(img)
                input_img = img.cuda()
                ret = model(input_img)
                did_outputs = ret[1]
                # did_outputs = ret[5]
                ff_d = ff_d + did_outputs
            # norm feature
            fnorm = torch.norm(ff_d, p=2, dim=1, keepdim=True)
            ff_d = ff_d.div(fnorm.expand_as(ff_d))
            ff_d = ff_d.detach().cpu().float()
            features = torch.cat((features, ff_d), 0)
        return features

    def extract_feature(model, dataloaders):
        features = torch.FloatTensor()
        for data in dataloaders:
            img, label = data
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
        return features

    def get_id(img_path):
        camera_id = []
        labels = []
        for path, v in img_path:
            # filename = path.split('/')[-1]
            filename = os.path.basename(path)
            label = filename[0:4]
            if 'msmt' in opt.test_dir:
                camera = filename[9:11]
            else:
                camera = filename.split('c')[1][0]
            if label[0:2] == '-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera))
        return camera_id, labels

    dataset_path = []
    for i in range(len(dataset_list)):
        dataset_path.append(image_datasets[dataset_list[i]].imgs)

    dataset_cam = []
    dataset_label = []
    for i in range(len(dataset_list)):
        cam, label = get_id(dataset_path[i])
        dataset_cam.append(cam)
        dataset_label.append(label)

    ######################################################################
    # Load Collected data Trained model
    print('---------test-----------')
    class_num = len(os.listdir(os.path.join(data_dir, 'train_all_new')))
    sid_num = class_num
    did_num = class_num * opt.domain_num
    did_embedding_net = ft_net(id_num=did_num)
    sid_embedding_net = ft_net(id_num=sid_num)
    model = DisentangleNet(did_embedding_net, sid_embedding_net)
    if use_gpu:
        model.cuda()
    if 'best' in opt.which_epoch or 'last' in opt.which_epoch:
        model = load_whole_network(model, name, opt.which_epoch + '_' + str(opt.net_loss_model))
    else:
        model = load_whole_network(model, name, opt.which_epoch)
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # Extract feature
    dataset_feature = []
    with torch.no_grad():
        for i in range(len(dataset_list)):
            dataset_feature.append(extract_feature(model, dataloaders[dataset_list[i]]))

    result = {'gallery_f': dataset_feature[0].numpy(), 'gallery_label': dataset_label[0], 'gallery_cam': dataset_cam[0],
              'query_f': dataset_feature[1].numpy(), 'query_label': dataset_label[1], 'query_cam': dataset_cam[1]}
    scipy.io.savemat('pytorch_result.mat', result)


if __name__ == '__main__':
    test_function(test_dir=opt.test_dir, net_loss_model=opt.net_loss_model, domain_num=opt.domain_num,
                  which_epoch=opt.which_epoch)
