import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################
# Contrastive loss
# Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
# --------------------------------------------------------------------
class ContrastiveLoss_CS(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss_CS, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output_c1, output_c2, output_s1, output_s2):
        # #Euclid distance
        distances_c = (output_c1 - output_c2).pow(2).sum(-1)
        distances_s = (output_s1 - output_s2).pow(2).sum(-1)
        losses_c = F.relu(self.margin - (distances_c + self.eps).sqrt()).pow(2)
        losses_s = distances_s
        return losses_c.mean(), losses_s.mean()

class ContrastiveLoss_diff(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss_diff, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output_c1, output_c2):
        # #Euclid distance
        # f_norm = output_c1.norm(p=2, dim=1, keepdim=True) + 1e-8
        # output_c1 = output_c1.div(f_norm)
        # f_norm = output_c2.norm(p=2, dim=1, keepdim=True) + 1e-8
        # output_c2 = output_c2.div(f_norm)
        distances_c = (output_c1 - output_c2).pow(2).sum(-1)
        losses_c = F.relu(self.margin - (distances_c + self.eps).sqrt()).pow(2)
        return losses_c.mean()


class ContrastiveLoss_same(nn.Module):
    def __init__(self):
        super(ContrastiveLoss_same, self).__init__()

    def forward(self, output_s1, output_s2):
        # #Euclid distance
        # f_norm = output_s1.norm(p=2, dim=1, keepdim=True) + 1e-8
        # output_s1 = output_s1.div(f_norm)
        # f_norm = output_s2.norm(p=2, dim=1, keepdim=True) + 1e-8
        # output_s2 = output_s2.div(f_norm)
        distances_s = (output_s1 - output_s2).pow(2).sum(-1)
        losses_s = distances_s
        return losses_s.mean()


class ContrastiveLoss_orth(nn.Module):
    def __init__(self):
        super(ContrastiveLoss_orth, self).__init__()

    def forward(self, output_o1, output_co2):
        # #Euclid distance
        # f_norm = output_c1.norm(p=2, dim=1, keepdim=True) + 1e-8
        # output_c1 = output_c1.div(f_norm)
        # f_norm = output_c2.norm(p=2, dim=1, keepdim=True) + 1e-8
        # output_c2 = output_c2.div(f_norm)
        distances_o = (output_o1 * output_co2).sum(-1).pow(2)
        losses_o = distances_o
        return losses_o.mean()

######################################################################
# Cross-entropy loss for soft-label
# --------------------------------------------------------------------
class SoftLabelLoss(nn.Module):
    def __init__(self):
        super(SoftLabelLoss, self).__init__()
        self.eps = 1e-9

    def forward(self, input, target, mask=None):
        if input.dim() > 2:  # N defines the number of images, C defines channels,  K class in total
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        maxRow, _ = torch.max(input.data, 1)  # outputs.data  return the index of the biggest value in each row
        maxRow = maxRow.unsqueeze(1)
        input.data = input.data - maxRow
        loss = self.loss_cross_entropy(input, target, mask)
        return loss

    def loss_cross_entropy(self, input_soft, target_soft, mask=None, reduce=True):
        input_soft = F.log_softmax(input_soft, dim=1)
        result = -target_soft * input_soft
        result = torch.sum(result, 1)
        if mask is not None:
            result = result * mask
        if reduce:
            if mask is not None:
                result = torch.mean(result) * (mask.shape[0] / (mask.sum() + self.eps))
            else:
                result = torch.mean(result)
        return result
