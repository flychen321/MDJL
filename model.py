import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
import os
from collections import OrderedDict


######################################################################
# Load parameters of model
# ---------------------------
def load_network(network, name, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'last')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load easy pretrained model: %s' % save_path)
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Load model
# ---------------------------
def load_whole_network(network, name, model_name=None):
    if model_name == None:
        save_path = os.path.join('./model', name, 'net_%s.pth' % 'whole_last')
    else:
        save_path = os.path.join('./model', name, 'net_%s.pth' % model_name)
    print('load whole pretrained model: %s' % save_path)
    net_original = torch.load(save_path)
    pretrained_dict = net_original.state_dict()
    model_dict = network.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network


######################################################################
# Save parameters of model
# ---------------------------
def save_network(network, name, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.state_dict(), save_path)


######################################################################
# Save model
# ---------------------------
def save_whole_network(network, name, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network, save_path)


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_out')
        init.constant_(m.bias.detach(), 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.detach(), 1.0, 0.02)
        init.constant_(m.bias.detach(), 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.detach(), std=0.001)
        init.constant_(m.bias.detach(), 0.0)


######################################################################
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
# --------------------------------------------------------------------
class Fc_ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=False, num_bottleneck=512):
        super(Fc_ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        f = x
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x, f


class Fc(nn.Module):
    def __init__(self, input_dim, relu=False, output_dim=512):
        super(Fc, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, output_dim)]
        add_block += [nn.BatchNorm1d(output_dim)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.add_block = add_block

    def forward(self, x):
        x = self.add_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim=512, class_num=751, dropout=0.5):
        super(ClassBlock, self).__init__()
        classifier = []
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        f = x
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x, f


# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, id_num):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.fc = Fc(2048)
        self.id_classifier = ClassBlock(class_num=id_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc(x)
        id_outputs, id_f = self.id_classifier(x)
        return id_outputs, id_f


######################################################################
# Define the DenseNet121-based Model
# --------------------------------------------------------------------
class ft_net_dense(nn.Module):
    def __init__(self, class_num=751, domain=3):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Fc(1024)
        self.classifier0 = ClassBlock(class_num=class_num)
        self.classifier1 = ClassBlock(class_num=class_num)
        self.classifier2 = ClassBlock(class_num=class_num)
        self.classifier3 = ClassBlock(class_num=class_num)
        self.classifier4 = ClassBlock(class_num=class_num)
        self.classifier5 = ClassBlock(class_num=class_num)
        self.classifier = [self.classifier0, self.classifier1, self.classifier2, self.classifier3, self.classifier4,
                           self.classifier5]
        self.domain = domain

    def forward(self, x):
        x = self.model.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.fc(x)
        outputs = torch.FloatTensor().cuda()
        features = torch.FloatTensor().cuda()
        for d in range(self.domain):
            output, feature = self.classifier[d](x)
            outputs = torch.cat((outputs, output), 0)
            features = torch.cat((features, feature), 0)
        return outputs, features

class DisentangleNet(nn.Module):
    def __init__(self, did_embedding_net, sid_embedding_net):
        super(DisentangleNet, self).__init__()
        self.did_embedding_net = did_embedding_net
        self.sid_embedding_net = sid_embedding_net

    def forward(self, x):
        did_outputs, did_f = self.did_embedding_net(x)
        sid_outputs, sid_f = self.sid_embedding_net(x)
        return did_outputs, did_f, sid_outputs, sid_f

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.classifier = Fc_ClassBlock(1024, 2, dropout=0.5, relu=False)

    # def forward(self, x, train=False):
    #     output, feature, mid_coder, feature_coder = self.embedding_net(x)
    #     if train == False:
    #         return feature
    #     part_len = feature.shape[0] // 4
    #     feature_1_0 = feature[part_len * 0:part_len * 1]
    #     feature_1_1 = feature[part_len * 1:part_len * 2]
    #     feature_2_0 = feature[part_len * 2:part_len * 3]
    #     feature_2_1 = feature[part_len * 3:part_len * 4]
    #
    #     feature1 = (feature_1_0 - feature_2_0).pow(2)
    #     feature2 = (feature_1_1 - feature_2_1).pow(2)
    #     features = torch.cat((feature1, feature2), 0)
    #     results = self.classifier.classifier(features)
    #     part_len = results.shape[0] // 2
    #     result1 = results[part_len * 0:part_len * 1]
    #     result2 = results[part_len * 1:part_len * 2]
    #     return output, \
    #            result1, result2, \
    #            feature_1_0, feature_1_1, feature_2_0, feature_2_1

    def forward(self, x1_0, x1_1=None, x2_0=None, x2_1=None):
        output_1_0, feature_1_0, mid_coder_1_0, feature_coder_1_0 = self.embedding_net(x1_0)
        if x1_1 is None:
            return feature_1_0
        output_1_1, feature_1_1, mid_coder_1_1, feature_coder_1_1 = self.embedding_net(x1_1)
        output_2_0, feature_2_0, mid_coder_2_0, feature_coder_2_0 = self.embedding_net(x2_0)
        output_2_1, feature_2_1, mid_coder_2_1, feature_coder_2_1 = self.embedding_net(x2_1)
        feature1 = (feature_1_0 - feature_2_0).pow(2)
        result1 = self.classifier.classifier(feature1)
        feature2 = (feature_1_1 - feature_2_1).pow(2)
        result2 = self.classifier.classifier(feature2)
        return output_1_0, output_1_1, output_2_0, output_2_1, \
               result1, result2, \
               feature_1_0, feature_1_1, feature_2_0, feature_2_1


######################################################################
# Auto_Encoder is used to disentangle different semantic feature
# self.embedding_net_a have the same network structure with self.embedding_net_p
# self.embedding_net_a is used for extracting appearance-related information
# self.embedding_net_p is used for extracting pose-related information
# self.decoder is used for reconstructing the original images
# --------------------------------------------------------------------
class Auto_Encoder_original(nn.Module):
    def __init__(self, embedding_net_a, embedding_net_p, decoder):
        super(Auto_Encoder_original, self).__init__()
        self.embedding_net_a = embedding_net_a
        self.embedding_net_p = embedding_net_p
        self.decoder = decoder

    def forward(self, x1_0, x1_1=None, x2_0=None, x2_1=None):
        output_a1_0, feature_a1_0, mid_coder_a1_0, feature_coder_a1_0 = self.embedding_net_a(x1_0)
        output_p1_0, feature_p1_0, mid_coder_p1_0, feature_coder_p1_0 = self.embedding_net_p(x1_0)
        if x1_1 is None:
            # return torch.cat((feature_a1_0, feature_p1_0), 1)
            return feature_a1_0
        output_a1_1, feature_a1_1, mid_coder_a1_1, feature_coder_a1_1 = self.embedding_net_a(x1_1)
        output_p1_1, feature_p1_1, mid_coder_p1_1, feature_coder_p1_1 = self.embedding_net_p(x1_1)
        output_a2_0, feature_a2_0, mid_coder_a2_0, feature_coder_a2_0 = self.embedding_net_a(x2_0)
        output_p2_0, feature_p2_0, mid_coder_p2_0, feature_coder_p2_0 = self.embedding_net_p(x2_0)
        output_a2_1, feature_a2_1, mid_coder_a2_1, feature_coder_a2_1 = self.embedding_net_a(x2_1)
        output_p2_1, feature_p2_1, mid_coder_p2_1, feature_coder_p2_1 = self.embedding_net_p(x2_1)

        mid = torch.cat((mid_coder_a1_0, mid_coder_p1_0), 1)
        rec_img_ap1010 = self.decoder(mid)
        mid = torch.cat((mid_coder_a1_1, mid_coder_p1_1), 1)
        rec_img_ap1111 = self.decoder(mid)
        mid = torch.cat((mid_coder_a2_0, mid_coder_p2_0), 1)
        rec_img_ap2020 = self.decoder(mid)
        mid = torch.cat((mid_coder_a2_1, mid_coder_p2_1), 1)
        rec_img_ap2121 = self.decoder(mid)

        mid = torch.cat((mid_coder_a1_0, mid_coder_p1_1), 1)
        rec_img_ap1011 = self.decoder(mid)
        mid = torch.cat((mid_coder_a1_1, mid_coder_p1_0), 1)
        rec_img_ap1110 = self.decoder(mid)
        mid = torch.cat((mid_coder_a2_0, mid_coder_p2_1), 1)
        rec_img_ap2021 = self.decoder(mid)
        mid = torch.cat((mid_coder_a2_1, mid_coder_p2_0), 1)
        rec_img_ap2120 = self.decoder(mid)

        mid = torch.cat((mid_coder_a1_0, mid_coder_p2_0), 1)
        rec_img_ap1020 = self.decoder(mid)
        mid = torch.cat((mid_coder_a1_1, mid_coder_p2_1), 1)
        rec_img_ap1121 = self.decoder(mid)
        mid = torch.cat((mid_coder_a2_0, mid_coder_p1_0), 1)
        rec_img_ap2010 = self.decoder(mid)
        mid = torch.cat((mid_coder_a2_1, mid_coder_p1_1), 1)
        rec_img_ap2111 = self.decoder(mid)

        mid = torch.cat((mid_coder_a1_0, mid_coder_p2_1), 1)
        rec_img_ap1021 = self.decoder(mid)
        mid = torch.cat((mid_coder_a1_1, mid_coder_p2_0), 1)
        rec_img_ap1120 = self.decoder(mid)
        mid = torch.cat((mid_coder_a2_0, mid_coder_p1_1), 1)
        rec_img_ap2011 = self.decoder(mid)
        mid = torch.cat((mid_coder_a2_1, mid_coder_p1_0), 1)
        rec_img_ap2110 = self.decoder(mid)

        return output_a1_0, output_a1_1, output_a2_0, output_a2_1, \
               feature_coder_a1_0, feature_coder_p1_0, feature_coder_a1_1, feature_coder_p1_1, \
               feature_coder_a2_0, feature_coder_p2_0, feature_coder_a2_1, feature_coder_p2_1, \
               rec_img_ap1010, rec_img_ap1111, rec_img_ap2020, rec_img_ap2121, \
               rec_img_ap1011, rec_img_ap1110, rec_img_ap2021, rec_img_ap2120, \
               rec_img_ap1020, rec_img_ap1121, rec_img_ap2010, rec_img_ap2111, \
               rec_img_ap1021, rec_img_ap1120, rec_img_ap2011, rec_img_ap2110


class Auto_Encoder_six(nn.Module):
    def __init__(self, embedding_net_a, embedding_net_p, decoder):
        super(Auto_Encoder_six, self).__init__()
        self.embedding_net_a = embedding_net_a
        self.embedding_net_p = embedding_net_p
        self.decoder = decoder
        self.verify = Fc_ClassBlock(1024, 2, dropout=0.5, relu=False)

    # def forward(self, x, train=False):
    #     output_a, feature_a, mid_coder_a, feature_coder_a = self.embedding_net_a(x)
    #     if train == False:
    #         return feature_a
    #     output_p, feature_p, mid_coder_p, feature_coder_p = self.embedding_net_p(x)
    #
    #     part_len = mid_coder_a.shape[0] // 3
    #
    #     mid_coder_a1_0 = mid_coder_a[part_len * 0:part_len * 1]
    #     mid_coder_a1_1 = mid_coder_a[part_len * 1:part_len * 2]
    #     mid_coder_a2_0 = mid_coder_a[part_len * 2:part_len * 3]
    #
    #     mid_coder_p1_0 = mid_coder_p[part_len * 0:part_len * 1]
    #     mid_coder_p1_1 = mid_coder_p[part_len * 1:part_len * 2]
    #     mid_coder_p2_0 = mid_coder_p[part_len * 2:part_len * 3]
    #
    #     feature_coder_a1_0 = feature_a[part_len * 0:part_len * 1]
    #     feature_coder_a1_1 = feature_a[part_len * 1:part_len * 2]
    #     feature_coder_a2_0 = feature_a[part_len * 2:part_len * 3]
    #
    #     feature_coder_p1_0 = feature_p[part_len * 0:part_len * 1]
    #     feature_coder_p1_1 = feature_p[part_len * 1:part_len * 2]
    #     feature_coder_p2_0 = feature_p[part_len * 2:part_len * 3]
    #
    #     mid_ap1010 = torch.cat((mid_coder_a1_0, mid_coder_p1_0), 1)
    #     mid_ap1111 = torch.cat((mid_coder_a1_1, mid_coder_p1_1), 1)
    #     mid_ap2020 = torch.cat((mid_coder_a2_0, mid_coder_p2_0), 1)
    #
    #     mid_ap1011 = torch.cat((mid_coder_a1_0, mid_coder_p1_1), 1)
    #     mid_ap1110 = torch.cat((mid_coder_a1_1, mid_coder_p1_0), 1)
    #
    #     mid_ap1020 = torch.cat((mid_coder_a1_0, mid_coder_p2_0), 1)
    #     mid_ap2010 = torch.cat((mid_coder_a2_0, mid_coder_p1_0), 1)
    #
    #     mid_ap1120 = torch.cat((mid_coder_a1_1, mid_coder_p2_0), 1)
    #     mid_ap2011 = torch.cat((mid_coder_a2_0, mid_coder_p1_1), 1)
    #     mid_all = torch.cat((mid_ap1010, mid_ap1111, mid_ap2020, \
    #                          mid_ap1011, mid_ap1110, \
    #                          mid_ap1020, mid_ap2010, \
    #                          mid_ap1120, mid_ap2011), 0)
    #     rec_img_all = self.decoder(mid_all)
    #     part_len = rec_img_all.shape[0] // 9
    #     rec_img_ap1010 = rec_img_all[part_len * 0: part_len * 1]
    #     rec_img_ap1111 = rec_img_all[part_len * 1: part_len * 2]
    #     rec_img_ap2020 = rec_img_all[part_len * 2: part_len * 3]
    #
    #     rec_img_ap1011 = rec_img_all[part_len * 3: part_len * 4]
    #     rec_img_ap1110 = rec_img_all[part_len * 4: part_len * 5]
    #
    #     rec_img_ap1020 = rec_img_all[part_len * 5: part_len * 6]
    #     rec_img_ap2010 = rec_img_all[part_len * 6: part_len * 7]
    #
    #     rec_img_ap1120 = rec_img_all[part_len * 7: part_len * 8]
    #     rec_img_ap2011 = rec_img_all[part_len * 8: part_len * 9]
    #
    #     return output_a, \
    #            feature_coder_a1_0, feature_coder_p1_0, feature_coder_a1_1, feature_coder_p1_1, \
    #            feature_coder_a2_0, feature_coder_p2_0, \
    #            rec_img_ap1010, rec_img_ap1111, rec_img_ap2020, \
    #            rec_img_ap1011, rec_img_ap1110, \
    #            rec_img_ap1020, rec_img_ap2010, \
    #            rec_img_ap1120, rec_img_ap2011

    def forward(self, x, train=False):
        output_a, feature_a, mid_coder_a, feature_coder_a = self.embedding_net_a(x)
        if train == False:
            return feature_a
        output_p, feature_p, mid_coder_p, feature_coder_p = self.embedding_net_p(x)

        part_len = mid_coder_a.shape[0] // 6

        mid_coder_a1_0 = mid_coder_a[part_len * 0:part_len * 1]
        mid_coder_a1_1 = mid_coder_a[part_len * 1:part_len * 2]
        mid_coder_a2_0 = mid_coder_a[part_len * 2:part_len * 3]
        mid_coder_a2_1 = mid_coder_a[part_len * 3:part_len * 4]
        mid_coder_a3_0 = mid_coder_a[part_len * 4:part_len * 5]
        mid_coder_a3_1 = mid_coder_a[part_len * 5:part_len * 6]

        mid_coder_p1_0 = mid_coder_p[part_len * 0:part_len * 1]
        mid_coder_p1_1 = mid_coder_p[part_len * 1:part_len * 2]
        mid_coder_p2_0 = mid_coder_p[part_len * 2:part_len * 3]
        mid_coder_p2_1 = mid_coder_p[part_len * 3:part_len * 4]
        mid_coder_p3_0 = mid_coder_p[part_len * 4:part_len * 5]
        mid_coder_p3_1 = mid_coder_p[part_len * 5:part_len * 6]

        feature_coder_a1_0 = feature_a[part_len * 0:part_len * 1]
        feature_coder_a1_1 = feature_a[part_len * 1:part_len * 2]
        feature_coder_a2_0 = feature_a[part_len * 2:part_len * 3]
        feature_coder_a2_1 = feature_a[part_len * 3:part_len * 4]
        feature_coder_a3_0 = feature_a[part_len * 4:part_len * 5]
        feature_coder_a3_1 = feature_a[part_len * 5:part_len * 6]

        feature_coder_p1_0 = feature_p[part_len * 0:part_len * 1]
        feature_coder_p1_1 = feature_p[part_len * 1:part_len * 2]
        feature_coder_p2_0 = feature_p[part_len * 2:part_len * 3]
        feature_coder_p2_1 = feature_p[part_len * 3:part_len * 4]
        feature_coder_p3_0 = feature_p[part_len * 4:part_len * 5]
        feature_coder_p3_1 = feature_p[part_len * 5:part_len * 6]

        feature1 = (feature_coder_a1_0 - feature_coder_a2_0).pow(2)
        feature2 = (feature_coder_a1_1 - feature_coder_a2_1).pow(2)

        feature3 = (feature_coder_a1_0 - feature_coder_a3_0).pow(2)
        feature4 = (feature_coder_a1_1 - feature_coder_a3_1).pow(2)

        feature5 = (feature_coder_a2_0 - feature_coder_a3_0).pow(2)
        feature6 = (feature_coder_a2_1 - feature_coder_a3_1).pow(2)

        features = torch.cat((feature1, feature2, feature3, feature4, feature5, feature6), 0)
        results = self.verify.classifier(features)
        part_len = results.shape[0] // 3
        result1 = results[part_len * 0:part_len * 1]
        result2 = results[part_len * 1:part_len * 2]
        result3 = results[part_len * 2:part_len * 3]

        mid_ap1010 = torch.cat((mid_coder_a1_0, mid_coder_p1_0), 1)
        mid_ap1111 = torch.cat((mid_coder_a1_1, mid_coder_p1_1), 1)
        mid_ap2020 = torch.cat((mid_coder_a2_0, mid_coder_p2_0), 1)
        mid_ap2121 = torch.cat((mid_coder_a2_1, mid_coder_p2_1), 1)
        mid_ap3030 = torch.cat((mid_coder_a3_0, mid_coder_p3_0), 1)
        mid_ap3131 = torch.cat((mid_coder_a3_1, mid_coder_p3_1), 1)

        mid_ap1011 = torch.cat((mid_coder_a1_0, mid_coder_p1_1), 1)
        mid_ap1110 = torch.cat((mid_coder_a1_1, mid_coder_p1_0), 1)
        mid_ap2021 = torch.cat((mid_coder_a2_0, mid_coder_p2_1), 1)
        mid_ap2120 = torch.cat((mid_coder_a2_1, mid_coder_p2_0), 1)
        mid_ap3031 = torch.cat((mid_coder_a3_0, mid_coder_p3_1), 1)
        mid_ap3130 = torch.cat((mid_coder_a3_1, mid_coder_p3_0), 1)

        mid_ap1020 = torch.cat((mid_coder_a1_0, mid_coder_p2_0), 1)
        mid_ap1121 = torch.cat((mid_coder_a1_1, mid_coder_p2_1), 1)
        mid_ap2010 = torch.cat((mid_coder_a2_0, mid_coder_p1_0), 1)
        mid_ap2111 = torch.cat((mid_coder_a2_1, mid_coder_p1_1), 1)

        mid_ap1021 = torch.cat((mid_coder_a1_0, mid_coder_p2_1), 1)
        mid_ap1120 = torch.cat((mid_coder_a1_1, mid_coder_p2_0), 1)
        mid_ap2011 = torch.cat((mid_coder_a2_0, mid_coder_p1_1), 1)
        mid_ap2110 = torch.cat((mid_coder_a2_1, mid_coder_p1_0), 1)
        mid_all = torch.cat((mid_ap1010, mid_ap1111, mid_ap2020, mid_ap2121, mid_ap3030, mid_ap3131, \
                             mid_ap1011, mid_ap1110, mid_ap2021, mid_ap2120, mid_ap3031, mid_ap3130, \
                             mid_ap1020, mid_ap1121, mid_ap2010, mid_ap2111, \
                             mid_ap1021, mid_ap1120, mid_ap2011, mid_ap2110), 0)
        rec_img_all = self.decoder(mid_all)
        part_len = rec_img_all.shape[0] // 20
        rec_img_ap1010 = rec_img_all[part_len * 0: part_len * 1]
        rec_img_ap1111 = rec_img_all[part_len * 1: part_len * 2]
        rec_img_ap2020 = rec_img_all[part_len * 2: part_len * 3]
        rec_img_ap2121 = rec_img_all[part_len * 3: part_len * 4]
        rec_img_ap3030 = rec_img_all[part_len * 4: part_len * 5]
        rec_img_ap3131 = rec_img_all[part_len * 5: part_len * 6]

        rec_img_ap1011 = rec_img_all[part_len * 6: part_len * 7]
        rec_img_ap1110 = rec_img_all[part_len * 7: part_len * 8]
        rec_img_ap2021 = rec_img_all[part_len * 8: part_len * 9]
        rec_img_ap2120 = rec_img_all[part_len * 9: part_len * 10]
        rec_img_ap3031 = rec_img_all[part_len * 10: part_len * 11]
        rec_img_ap3130 = rec_img_all[part_len * 11: part_len * 12]

        rec_img_ap1020 = rec_img_all[part_len * 12: part_len * 13]
        rec_img_ap1121 = rec_img_all[part_len * 13: part_len * 14]
        rec_img_ap2010 = rec_img_all[part_len * 14: part_len * 15]
        rec_img_ap2111 = rec_img_all[part_len * 15: part_len * 16]

        rec_img_ap1021 = rec_img_all[part_len * 16: part_len * 17]
        rec_img_ap1120 = rec_img_all[part_len * 17: part_len * 18]
        rec_img_ap2011 = rec_img_all[part_len * 18: part_len * 19]
        rec_img_ap2110 = rec_img_all[part_len * 19: part_len * 20]

        return output_a, output_p, \
               result1, result2, result3, \
               feature_coder_a1_0, feature_coder_p1_0, feature_coder_a1_1, feature_coder_p1_1, \
               feature_coder_a2_0, feature_coder_p2_0, feature_coder_a2_1, feature_coder_p2_1, \
               feature_coder_a3_0, feature_coder_p3_0, feature_coder_a3_1, feature_coder_p3_1, \
               rec_img_ap1010, rec_img_ap1111, rec_img_ap2020, rec_img_ap2121, rec_img_ap3030, rec_img_ap3131, \
               rec_img_ap1011, rec_img_ap1110, rec_img_ap2021, rec_img_ap2120, rec_img_ap3031, rec_img_ap3130, \
               rec_img_ap1020, rec_img_ap1121, rec_img_ap2010, rec_img_ap2111, \
               rec_img_ap1021, rec_img_ap1120, rec_img_ap2011, rec_img_ap2110

    # def forward(self, x1_0, x1_1=None, x2_0=None, x2_1=None):
    #     output_a1_0, feature_a1_0, mid_coder_a1_0, feature_coder_a1_0 = self.embedding_net_a(x1_0)
    #     if x1_1 is None:
    #         return feature_a1_0
    #     output_a1_1, feature_a1_1, mid_coder_a1_1, feature_coder_a1_1 = self.embedding_net_a(x1_1)
    #     output_a2_0, feature_a2_0, mid_coder_a2_0, feature_coder_a2_0 = self.embedding_net_a(x2_0)
    #     output_a2_1, feature_a2_1, mid_coder_a2_1, feature_coder_a2_1 = self.embedding_net_a(x2_1)
    #
    #     return output_a1_0, output_a1_1, output_a2_0, output_a2_1


#############################################################################################################
# decoder is used for reconstructing the original images, ant it consists 5 transposed convolutional layers
# The input size is 2048*8*4
# The output size is 3*256*128
# -----------------------------------------------------------------------------------------------------------
class decoder(nn.Module):
    def __init__(self, in_channels=2048):
        super(decoder, self).__init__()
        layer0 = nn.Sequential(OrderedDict([
            ('conv0',
             nn.ConvTranspose2d(in_channels, 512, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(512)),
            ('relu0', nn.LeakyReLU(0.1)),
            ('drop0', nn.Dropout(p=0.5)),
        ]))
        layer1 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(128)),
            ('relu0', nn.LeakyReLU(0.1)),
            ('drop0', nn.Dropout(p=0.5)),
        ]))
        layer2 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.LeakyReLU(0.1)),
            ('drop0', nn.Dropout(p=0.5)),
        ]))
        layer3 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.LeakyReLU(0.1)),
            ('drop0', nn.Dropout(p=0.5)),
        ]))
        layer4 = nn.Sequential(OrderedDict([
            ('conv0', nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(3)),
        ]))
        layer0.apply(weights_init_kaiming)
        layer1.apply(weights_init_kaiming)
        layer2.apply(weights_init_kaiming)
        layer3.apply(weights_init_kaiming)
        layer4.apply(weights_init_kaiming)

        self.layer0 = layer0
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
