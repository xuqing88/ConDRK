import torch
from torch import nn
from .memory import ContrastMemory

eps = 1e-7


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        # self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        # self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.criterion_t = ContrastLoss(opt.n_data)
        self.criterion_s = ContrastLoss(opt.n_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        # f_s = self.embed_s(f_s)
        # f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class ContrastLoss_self(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss_self, self).__init__()
        self.n_data = n_data
        self.l2norm = Normalize(2)
        # self.embed = Embed(dim_in=6, dim_out=6)

    def forward(self, x, x_pos, x_neg):
        # tau =0.07
        tau = 0.05
        # x = self.embed(x)
        # x_pos = self.l2norm(x_pos)
        # x_neg = self.l2norm(x_neg)

        bsz, inputSize = x.shape[0], x.shape[1]

        K = x_pos.shape[1]
        K_neg = x_neg.shape[1]

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        # P_pos = torch.bmm(x.view(bsz, 1, inputSize), x_pos.view(bsz, inputSize, 1))
        # P_pos = torch.exp(P_pos / tau).squeeze(dim=2)

        P_pos = torch.nn.functional.cosine_similarity(x, x_pos).unsqueeze(dim=1)
        # x = torch.log_softmax(x, dim=1)
        # x_pos = torch.softmax(x_pos, dim=1)
        # P_pos = torch.nn.functional.kl_div(x,x_pos,reduction='none').mean(dim=1).unsqueeze(dim=1)
        # P_pos = torch.nn.functional.mse_loss(x, x_pos, reduction='none').mean(dim=1).unsqueeze(dim=1)

        P_pos = torch.exp(P_pos/tau)
        log_D1 = torch.div(P_pos, P_pos.add(K * Pn + eps)).log_()

        # loss for K negative pair
        # P_neg = torch.bmm(x_neg, x.view(bsz, inputSize, 1))
        # P_neg = torch.exp(P_neg/tau).squeeze(dim=2)
        output = []
        for i in range(K_neg):
            sim_neg = torch.nn.functional.cosine_similarity(x, x_neg[:, i, :]).unsqueeze(dim=1)
            # temp = torch.softmax(x_neg[:, i, :], dim=1)
            # sim_neg =torch.nn.functional.kl_div(x,temp,reduction='none').mean(dim=1).unsqueeze(dim=1)
            # sim_neg = torch.nn.functional.mse_loss(x, x_neg[:, i, :], reduction='none').mean(dim=1).unsqueeze(dim=1)
            output.append(sim_neg)
        P_neg = torch.cat([x for x in output], dim=1).reshape((bsz, K_neg, 1))
        P_neg = torch.exp(P_neg / tau)
        log_D0 = torch.div(P_neg.clone().fill_(K_neg * Pn), P_neg.add(K_neg * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class ContrastLoss_self_v2(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss_self_v2, self).__init__()
        self.n_data = n_data
        self.kl_div = nn.KLDivLoss(reduction="none")
        self.contrast = ContrastLoss(self.n_data)

    def forward(self, x, x_pos, x_neg):

        bsz, inputSize = x.shape[0], x.shape[1]
        K = x_neg.shape[1]
        sim_pos = torch.nn.functional.cosine_similarity(x, x_pos).unsqueeze(dim=1)

        # log_p = torch.log_softmax(x, dim=1)
        # q = torch.softmax(x_pos, dim=1)
        # sim_pos = torch.exp(self.kl_div(log_p, q)).sum(dim=1).unsqueeze(dim=1)

        output = [sim_pos]
        for i in range(K):
            q = torch.softmax(x_neg[:,i,:], dim=1)
            sim_neg = torch.nn.functional.cosine_similarity(x, q).unsqueeze(dim=1)
            # sim_neg = torch.exp(self.kl_div(log_p, q)).sum(dim=1).unsqueeze(dim=1)
            output.append(sim_neg)

        output = torch.cat([x for x in output], dim=1).reshape((bsz,K+1,1))
        loss = self.contrast(output)

        return loss
