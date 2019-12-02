import os, sys

lib_path = os.path.abspath(os.path.join('..', 'datasets'))
sys.path.append(lib_path)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from utils import one_hot_embedding

def one_hot_embedding(labels, num_classes):
    """
    Embedding labels to one-hot form.
    Args:
    :param labels: (LongTensor) class label, sized [N,].
    :param num_classes: (int) number of classes.
    :return:
            (tensor) encoded labels, size [N, #classes].
    """
    y = torch.eye(num_classes)  # [D, D]
    return y[labels]  # [N, D]

class FocalLoss(nn.Module):
    def __init__(self, num_classes=80):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        """
        Focal loss.
        Args:
        :param x: (tensor) sized [N, D]
        :param y: (tensor) sized [N, ].
        :return:
                (tensor) focal loss.
        """
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu().long(), 1 + self.num_classes)  # [N, 81]
        t = t[:, 1:]  # exclude background
        t = Variable(t).cuda()

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)
        w = alpha * t + (1 - alpha) * (1 - t)
        w = w * (1 - pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w.detach(), size_average=False)

    def focal_loss_alt(self, x, y):
        """
        Focal loss alternative.
        Args:
        :param x: (tensor) sized [N, D]
        :param y: (tensor) sized [N, ].
        :return:
                (tensor) focal loss.
        """
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu().long(), 1 + self.num_classes)  # [N, 81]
        t = t[:, 1:]  # exclude background
        t = Variable(t).cuda()

        xt = x * (2 * t - 1)  # xt = x if t>0 else -x
        pt = (2 * xt + 1).sigmoid()

        w = alpha * t + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum()

    @staticmethod
    def where(cond, x_1, x_2):
        return (cond.float() * x_1) + ((1 - cond.float()) * x_2)

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        """
        Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
        :param loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
        :param loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
        :param cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
        :param cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
            (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        """
        # print(cls_targets)
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N, #anchors]
        num_pos = pos.data.long().sum()
        # print(num_pos, 'num_pos')

        ##########################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ##########################################################
        if num_pos > 0:
            mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N, #anchors, 4]
            masked_loc_preds = loc_preds[mask].view(-1, 4)  # [#pos, 4]
            masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos, 4]
            # loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
            regression_diff = torch.abs(masked_loc_targets - masked_loc_preds)
            loc_loss = self.where(torch.le(regression_diff, 1.0 / 9.0), 0.5 * 9.0 * torch.pow(regression_diff, 2),
                                  regression_diff - 0.5 / 9.0)
            # use mean() here, so the loc_loss dont have to divide num_pos
            # loc_loss = loc_loss.sum()
            loc_loss = loc_loss.mean()
        else:
            num_pos = 1.
            loc_loss = Variable(torch.Tensor([0]).float().cuda())

        ##########################################################
        # cls_loss = FocalLoss(cls_preds, cls_targets)
        ##########################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])

        # print('loc_loss: {:.3f} | cls_loss: {:.3f}'.format(loc_loss.data[0] / num_pos, cls_loss.data[0] / num_pos),
        #       end=' | ')
        # loss = (loc_loss + cls_loss) / num_pos
        loc_loss = loc_loss
        cls_loss = cls_loss / num_pos
        return loc_loss, cls_loss