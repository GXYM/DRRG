import torch
import torch.nn as nn
import torch.nn.functional as F


class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def ohem(predict, target, train_mask, negative_ratio=3.):
        pos = (target * train_mask).byte()
        neg = ((1 - target) * train_mask).byte()

        n_pos = pos.float().sum()

        if n_pos.item() > 0:
            loss_pos = F.cross_entropy(predict[pos], target[pos], reduction='sum')
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))
        else:
            loss_pos = torch.tensor(0.)
            loss_neg = F.cross_entropy(predict[neg], target[neg], reduction='none')
            n_neg = 100
        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    @staticmethod
    def smooth_l1_loss(inputs, target, sigma=9.0):
        try:
            diff = torch.abs(inputs - target)
            less_one = (diff < 1.0 / sigma).float()
            loss = less_one * 0.5 * diff ** 2 * sigma \
                   + torch.abs(torch.tensor(1.0) - less_one) * (diff - 0.5 / sigma)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            loss = torch.tensor(0.0)

        return loss

    def gcn_loss(self, gcn_data):
        # gcn loss
        gcn_pred = gcn_data[0]
        labels = gcn_data[1].view(-1).long()
        loss = F.cross_entropy(gcn_pred, labels)  # *torch.tensor(0.0)

        return loss

    def forward(self, inputs, gcn_data, train_mask, tr_mask, tcl_mask, radii_map, sin_map, cos_map):
        """
        calculate textsnake loss
        :param inputs: (Variable), network predict, (BS, 8, H, W)
        :param gcn_data: (Variable), (gcn_pred ,gtmat_batch)
        :param tr_mask: (Variable), TR target, (BS, H, W)
        :param tcl_mask: (Variable), TCL target, (BS, H, W)
        :param sin_map: (Variable), sin target, (BS, H, W)
        :param cos_map: (Variable), cos target, (BS, H, W)
        :param radii_map: (Variable), radius target, (BS, H, W)
        :param train_mask: (Variable), training mask, (BS, H, W)
        :return: loss_tr, loss_tcl, loss_radii, loss_sin, loss_cos
        """

        tr_pred = inputs[:, :2].permute(0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)
        tcl_pred = inputs[:, 2:4].permute(0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)
        sin_pred = inputs[:, 4].contiguous().view(-1)  # (BSxHxW,)
        cos_pred = inputs[:, 5].contiguous().view(-1)  # (BSxHxW,)

        # regularize sin and cos: sum to 1
        scale = torch.sqrt(1.0 / (sin_pred ** 2 + cos_pred ** 2 + 0.0001))
        sin_pred = sin_pred * scale
        cos_pred = cos_pred * scale

        top_pred = inputs[:, 6].contiguous().view(-1)  # (BSxHxW,)
        bot_pred = inputs[:, 7].contiguous().view(-1)  # (BSxHxW,)

        train_mask = train_mask.contiguous().view(-1)  # (BSxHxW,)

        tr_mask = tr_mask.contiguous().view(-1)
        tcl_mask = tcl_mask[:, :, :, 0].contiguous().view(-1)
        sin_map = sin_map.contiguous().view(-1)
        cos_map = cos_map.contiguous().view(-1)
        top_map = radii_map[:, :, :, 0].contiguous().view(-1)
        bot_map = radii_map[:, :, :, 1].contiguous().view(-1)

        # loss_tr = F.cross_entropy(tr_pred[train_mask], tr_mask[train_mask].long())
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())

        loss_tcl = torch.tensor(0.)
        tr_train_mask = train_mask * tr_mask
        tr_neg_mask = 1 - tr_train_mask
        if tr_train_mask.sum().item() > 0:
            loss_tcl_pos = F.cross_entropy(tcl_pred[tr_train_mask], tcl_mask[tr_train_mask].long())
            loss_tcl_neg = F.cross_entropy(tcl_pred[tr_neg_mask], tcl_mask[tr_neg_mask].long())
            loss_tcl = loss_tcl_pos #+ loss_tcl_neg

        # geometry losses
        loss_radii = torch.tensor(0.)
        loss_sin = torch.tensor(0.)
        loss_cos = torch.tensor(0.)
        tcl_train_mask = train_mask * tcl_mask
        if tcl_train_mask.sum().item() > 0:
            ones = torch.ones_like(top_pred[tcl_mask]).float()

            loss_top = F.smooth_l1_loss(top_pred[tcl_mask] / (top_map[tcl_mask]+0.01), ones, reduction='none')
            loss_bot = F.smooth_l1_loss(bot_pred[tcl_mask] / (bot_map[tcl_mask]+0.01), ones, reduction='none')

            rad_map = top_map[tcl_mask] + bot_map[tcl_mask]
            # loss_radii = torch.mean(torch.log10(rad_map+1.0)*(loss_top+loss_bot))
            loss_radii = torch.mean(loss_top + loss_bot)

            # loss_radii=torch.tensor(0);
            loss_sin = self.smooth_l1_loss(sin_pred[tcl_mask], sin_map[tcl_mask])
            loss_cos = self.smooth_l1_loss(cos_pred[tcl_mask], cos_map[tcl_mask])

        # ##  Graph convolution loss
        gcn_loss = self.gcn_loss(gcn_data)
        # gcn_loss = torch.tensor(0.)

        return loss_tr, loss_tcl, loss_sin, loss_cos, loss_radii, gcn_loss

