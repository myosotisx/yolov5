import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# AB Distillation

class AB_Distill_Normal2Tiny(nn.Module):

    def __init__(self, t_net, s_net, batch_size, loss_multiplier):
        super(AB_Distill_Normal2Tiny, self).__init__()

        self.batch_size = batch_size
        self.loss_multiplier = loss_multiplier

        # Connector function
        C1 = [nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(256)]
        C2 = [nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(512)]
        # C3 = [nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(1024)]

        # Initialize connector
        # for m in C1 + C2 + C3:
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        for m in C1 + C2:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.Connect1 = nn.Sequential(*C1)
        self.Connect2 = nn.Sequential(*C2)
        # self.Connect3 = nn.Sequential(*C3)

        # self.Connectors = nn.ModuleList([self.Connect1, self.Connect2, self.Connect3])
        self.Connectors = nn.ModuleList([self.Connect1, self.Connect2])

        self.t_net = t_net
        self.s_net = s_net

    def criterion_active_L2(self, source, target, margin):
        loss = ((source + margin) ** 2 * ((source > -margin) & (target <= 0)).float() +
                (source - margin) ** 2 * ((source <= margin) & (target > 0)).float())
        return torch.abs(loss).sum()

    def forward(self, x):
        inputs = x

        # Teacher network
        with torch.no_grad():
            res1_t = self.t_net.model[0:5](inputs)
            res2_t = self.t_net.model[5:7](res1_t)
            # res3_t = self.t_net.model[7:10](res2_t)

        # Student network
        res1_s = self.s_net.model[0:5](inputs)
        res2_s = self.s_net.model[5:7](res1_s)
        # res3_s = self.s_net.model[7:10](res2_s)

        # Simularity
        # sim_g3 = 1-((self.Connect3(res3_s) > 0) ^ (res3_t.detach() > 0)).sum().float() / res3_t.nelement()
        sim_g2 = 1-((self.Connect2(res2_s) > 0) ^ (res2_t.detach() > 0)).sum().float() / res2_t.nelement()
        sim_g1 = 1-((self.Connect1(res1_s) > 0) ^ (res1_t.detach() > 0)).sum().float() / res1_t.nelement()

        # Alternative loss
        margin = 1.0
        loss_g2 = self.criterion_active_L2(self.Connect2(res2_s), res2_t.detach(), margin) / self.batch_size / 2 / 1000
        loss_g1 = self.criterion_active_L2(self.Connect1(res1_s), res1_t.detach(), margin) / self.batch_size / 4 / 1000

        loss = loss_g1+loss_g2

        loss *= self.loss_multiplier

        # Return all losses
        # return loss, loss_g1, loss_g2, loss_g3, sim_g1, sim_g2, sim_g3
        return loss, loss_g1, loss_g2, 0, sim_g1, sim_g2, 0


class AB_Distill_Normal2Tiny2(nn.Module):

    def __init__(self, t_net, s_net, batch_size, loss_multiplier):
        super(AB_Distill_Normal2Tiny2, self).__init__()

        self.batch_size = batch_size
        self.loss_multiplier = loss_multiplier

        # Connector function
        C1 = [nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(256)]
        # C2 = [nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(512)]
        # C3 = [nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(1024)]

        # Initialize connector
        for m in C1:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.Connect1 = nn.Sequential(*C1)
        # self.Connect2 = nn.Sequential(*C2)
        # self.Connect3 = nn.Sequential(*C3)

        self.Connectors = nn.ModuleList([self.Connect1])

        self.t_net = t_net
        self.s_net = s_net

    def criterion_active_L2(self, source, target, margin):
        loss = ((source + margin) ** 2 * ((source > -margin) & (target <= 0)).float() +
                (source - margin) ** 2 * ((source <= margin) & (target > 0)).float())
        return torch.abs(loss).sum()

    def forward(self, x):
        inputs = x

        # Teacher network
        with torch.no_grad():
            res1_t = self.t_net.model[0:5](inputs)
        # res2_t = self.t_net.model[7:9](res1_t)
        # res3_t = self.t_net.model[9:11](res2_t)

        # Student network
        res1_s = self.s_net.model[0:5](inputs)
        # res2_s = self.s_net.model[7:9](res1_s)
        # res3_s = self.s_net.model[9:11](res2_s)

        # Activation transfer loss
        # loss_AT3 = ((self.Connect3(res3_s) > 0) ^ (res3_t > 0)).sum().float() / res3_t.nelement()
        # loss_AT2 = ((self.Connect2(res2_s) > 0) ^ (res2_t > 0)).sum().float() / res2_t.nelement()
        sim_g1 = ((self.Connect1(res1_s) > 0) ^ (res1_t > 0)).sum().float() / res1_t.nelement()

        # Alternative loss
        margin = 1.0
        # loss = self.criterion_active_L2(self.Connect3(res3_s), res3_t.detach(), margin) / self.batch_size
        # loss += self.criterion_active_L2(self.Connect2(res2_s), res2_t.detach(), margin) / self.batch_size / 2
        # loss = self.criterion_active_L2(self.Connect2(res2_s), res2_t.detach(), margin) / self.batch_size / 2
        # loss += self.criterion_active_L2(self.Connect1(res1_s), res1_t.detach(), margin) / self.batch_size / 4
        loss_g1 = self.criterion_active_L2(self.Connect1(res1_s), res1_t.detach(), margin) / self.batch_size / 4 / 1000

        loss = loss_g1

        loss *= self.loss_multiplier

        # Return all losses
        return loss, loss_g1, 0, 0, sim_g1, 0, 0


class AT_Distill_Normal2Tiny(nn.Module):

    def __init__(self, t_net, s_net, beta):
        super(AT_Distill_Normal2Tiny, self).__init__()
        
        self.t_net = t_net
        self.s_net = s_net

        self.beta = beta

    def forward(self, x, res_s):
        inputs = x
        
        # Teacher network
        with torch.no_grad():
            res1_t = self.t_net.model[0:5](inputs)
            res2_t = self.t_net.model[5:7](res1_t)
        # res3_t = self.t_net.model[9:11](res2_t)

        # Student network
        res1_s = res_s[0]
        res2_s = res_s[1]
        # res3_s = self.s_net.model[9:11](res2_s)

        # Attention loss
        loss_group1 = self.at_loss(res1_t.detach(), res1_s)*self.beta
        loss_group2 = self.at_loss(res2_t.detach(), res2_s)*self.beta
        # loss_group3 = self.at_loss(res3_t.detach(), res3_s)
        # loss = loss_group1+loss_group2+loss_group3
        loss = loss_group1+loss_group2

        # return loss, loss_group1, loss_group2, loss_group3
        return loss, torch.Tensor([loss, loss_group1, loss_group2, 0]).cuda().detach()


    def at(self, x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def at_loss(self, x, y):
        return (self.at(x) - self.at(y)).pow(2).mean()


class AT_Distill_Normal2Tiny2(nn.Module):

    def __init__(self, t_net, s_net, alpha=4e-4, beta=2e-2, T=0.5):
        super(AT_Distill_Normal2Tiny2, self).__init__()
        
        self.t_net = t_net
        self.s_net = s_net

        self.alpha = alpha
        self.beta = beta
        self.T = T

    def forward(self, x, res_s):
        inputs = x
        
        # Teacher network
        with torch.no_grad():
            res1_t = self.t_net.model[0:5](inputs)
            res2_t = self.t_net.model[5:7](res1_t)
        # res3_t = self.t_net.model[9:11](res2_t)

        # Student network
        res1_s = res_s[0]
        res2_s = res_s[1]
        # res3_s = self.s_net.model[9:11](res2_s)

        # Attention loss
        loss_group1 = self.at_loss(res1_s, res1_t.detach())
        loss_group2 = self.at_loss(res2_s, res2_t.detach())
        # loss_group3 = self.at_loss(self.Connect3(res3_s), res3_t.detach())
        # loss = loss_group1+loss_group2+loss_group3
        loss = loss_group1+loss_group2

        # return loss, loss_group1, loss_group2, loss_group3
        return loss, torch.Tensor([loss, loss_group1, loss_group2, 0]).cuda().detach()

    def gs(self, x, power=2):
        # Spatial attention map
        return F.normalize(x.pow(power).mean(1).view(x.size(0), -1))

    def at_loss(self, x, y):
        gsx = self.gs(x)
        gsy = self.gs(y)

        tmp = (gsx-gsy).pow(2)

        # Loss AT
        # Attention loss
        lat = tmp.sum()

        # Loss AM
        # Attention mask loss to emphasize the learning of foreground area
        ms = F.softmax((gsx+gsy)/self.T, dim=1)
        lam = (tmp * ms).sum()

        return self.alpha * lat + self.beta * lam
