import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class ABD_Module_YOLOv5(nn.Module):

    """
        Activation Boundaries Distillation for YOLOv5
        https://ojs.aaai.org/index.php/AAAI/article/view/4264/4142
    """
    def __init__(self, t_net, s_net, margin=1.0, loss_multiplier=1.0, g3_enable=False):
        super(ABD_Module_YOLOv5, self).__init__()

        # Initialize basic parameters
        self.t_net = t_net
        self.s_net = s_net

        self.margin = margin
        self.loss_multiplier = loss_multiplier
        # Flag for distilling g3 feature
        self.g3_enable = g3_enable

        # Setup connector functions(adaptive layers)
        c1 = [nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(256)]
        c2 = [nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(512)]
        
        self.c1 = nn.Sequential(*c1)
        self.c2 = nn.Sequential(*c2)

        if g3_enable:
            c3 = [nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(1024)]
            self.c3 = nn.Sequential(*c3)
            self.modules_list = nn.ModuleList([self.c1, self.c2, self.c3])
        else:
            self.modules_list = nn.ModuleList([self.c1, self.c2])

        initialize_weights(self.modules_list.modules())

        # Assign headers for print
        if self.g3_enable:
            self.headers = ('loss', 'lab_g1', 'lab_g2', 'lab_g3', 'sim_g1', 'sim_g2', 'sim_g3')
        else:
            self.headers = ('loss', 'lab_g1', 'lab_g2', 'sim_g1', 'sim_g2')

    def criterion_active_L2(self, source, target, margin):
        loss = ((source + margin) ** 2 * ((source > -margin) & (target <= 0)).float() +
                (source - margin) ** 2 * ((source <= margin) & (target > 0)).float())
        return torch.abs(loss).sum()

    def forward(self, x):
        inputs = x

        # Teacher network forward
        with torch.no_grad():
            res1_t = self.t_net.model[0:5](inputs)
            res2_t = self.t_net.model[5:7](res1_t)
            if self.g3_enable:
                res3_t = self.t_net.model[7:10](res2_t)

        # Student network forward
        res1_s = self.s_net.model[0:5](inputs)
        res2_s = self.s_net.model[5:7](res1_s)
        if self.g3_enable:
            res3_s = self.s_net.model[7:10](res2_s)

        # Simularity
        sim_g1 = 1-((self.c1(res1_s) > 0) ^ (res1_t.detach() > 0)).sum().float() / res1_t.nelement()
        sim_g2 = 1-((self.c2(res2_s) > 0) ^ (res2_t.detach() > 0)).sum().float() / res2_t.nelement()
        if self.g3_enable:
            sim_g3 = 1-((self.c3(res3_s) > 0) ^ (res3_t.detach() > 0)).sum().float() / res3_t.nelement()
        
        # Alternative loss
        margin = 1.0
        b = res1_t.size(0)
        
        lab_g1 = self.criterion_active_L2(self.c1(res1_s), res1_t.detach(), margin) / b / 4 / 1000
        lab_g2 = self.criterion_active_L2(self.c2(res2_s), res2_t.detach(), margin) / b / 2 / 1000

        if self.g3_enable:
            lab_g3 = self.criterion_active_L2(self.c3(res3_s), res3_t.detach(), margin) / b / 1000
            loss = lab_g1+lab_g2+lab_g3
            loss *= self.loss_multiplier
            return loss, torch.Tensor([loss, lab_g1, lab_g2, lab_g3, sim_g1, sim_g2, sim_g3]).cuda().detach()
        else:
            loss = lab_g1+lab_g2
            loss *= self.loss_multiplier
            return loss, torch.Tensor([loss, lab_g1, lab_g2, sim_g1, sim_g2]).cuda().detach()


class AGD_Module_YOLOv5(nn.Module):

    """
        Attention-Guided Distillation Module for YOLOv5
        https://openreview.net/pdf/1e6969024e2fab8681c02ff62a2dbfc4feedcff4.pdf
    """
    def __init__(self, t_net, s_net, alpha=4e-4, beta=2e-2, T=0.5, channel_enable=True):
        super(AGD_Module_YOLOv5, self).__init__()
        
        # Initialize basic parameters
        self.t_net = t_net
        self.s_net = s_net

        self.alpha = alpha
        self.beta = beta
        self.T = T
        # Flag for utilizing channel mask
        self.channel_enable = channel_enable

        # Setup connector functions(adaptive layers)
        c1 = [nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),]
        c2 = [nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),]

        self.c1 = nn.Sequential(*c1)
        self.c2 = nn.Sequential(*c2)

        if self.channel_enable:
            c1c = [nn.Linear(128, 256, bias=False),]
            c2c = [nn.Linear(256, 512, bias=False),]

            self.c1c = nn.Sequential(*c1c)
            self.c2c = nn.Sequential(*c2c)

            self.modules_list = nn.ModuleList([self.c1, self.c2, self.c1c, self.c2c])
        else:
            self.modules_list = nn.ModuleList([self.c1, self.c2])

        initialize_weights(self.modules_list.modules())

        # Assign headers for print
        if channel_enable:
            self.headers = ('loss', 'lats_g1', 'lats_g2', 'latc_g1', 'latc_g2', 'lam_g1', 'lam_g2')
        else:
            self.headers = ('loss', 'lats_g1', 'lats_g2', 'lam_g1', 'lam_g2')

    def forward(self, x, res_s=None):
        inputs = x
        
        # Teacher network forward
        with torch.no_grad():
            res1_t = self.t_net.model[0:5](inputs)
            res2_t = self.t_net.model[5:7](res1_t)

        # Student network forward
        if res_s is None:
            res1_s = self.s_net.model[0:5](inputs)
            res2_s = self.s_net.model[5:7](res1_s)
        else:
            res1_s = res_s[0]
            res2_s = res_s[1]

        if self.channel_enable:
            # Loss calculation w/ channel mask
            lats_g1, latc_g1, lam_g1 = self.at_loss(res1_t.detach(), res1_s, self.c1, self.c1c)
            lats_g2, latc_g2, lam_g2 = self.at_loss(res2_t.detach(), res2_s, self.c2, self.c2c)

            lats_g1, latc_g1, lam_g1 = lats_g1 * self.alpha, latc_g1 * self.alpha, lam_g1 * self.beta
            lats_g2, latc_g2, lam_g2 = lats_g2 * self.alpha, latc_g2 * self.alpha, lam_g2 * self.beta
            loss = lats_g1+latc_g1+lam_g1+lats_g2+latc_g2+lam_g2

            return loss, torch.Tensor([loss, lats_g1, lats_g2, latc_g1, latc_g2, lam_g1, lam_g2]).cuda().detach()
        else:
            # Loss calculation w/o channel mask
            lats_g1, lam_g1 = self.at_loss(res1_t.detach(), res1_s, self.c1)
            lats_g2, lam_g2 = self.at_loss(res2_t.detach(), res2_s, self.c2)

            lats_g1, lam_g1 = lats_g1 * self.alpha, lam_g1 * self.beta
            lats_g2, lam_g2 = lats_g2 * self.alpha, lam_g2 * self.beta
            loss = lats_g1+lam_g1+lats_g2+lam_g2

            return loss, torch.Tensor([loss, lats_g1, lats_g2, lam_g1, lam_g2]).cuda().detach()


    def gs(self, x):
        # Spatial attention map
        return x.pow(2).mean(dim=1)

    def gc(self, x):
        # Channel attention vector
        return x.view(x.size(0), x.size(1), -1).pow(2).mean(dim=2)

    def at_loss(self, x, y, connector=None, projc=None):
        # x represents teacher's feature, y represents student's feature
        b, c, h, w = x.size()

        # loss AT
        gsx = self.gs(x)
        gsy = self.gs(y)
        lats = (gsx-gsy).pow(2).sum() / b

        if self.channel_enable:
            gcx = self.gc(x)
            gcy = self.gc(y) if projc is None else projc(self.gc(y))
            latc = (gcx-gcy).pow(2).sum() / b

        # Loss AM
        x_ = x.view(x.size(0), x.size(1), -1)
        y_ = y.view(x.size(0), x.size(1), -1) if connector is None else connector(y).view(x.size(0), x.size(1), -1)

        gsx = gsx.view(gsx.size(0), -1)
        gsy = gsy.view(gsy.size(0), -1)
        
        ms = h * w * F.softmax(gsx/self.T, dim=1)
        ms = ms.unsqueeze(dim=1).repeat(1, c, 1)

        if self.channel_enable:
            mc = c * F.softmax(gcx/self.T, dim=1)
            mc = mc.unsqueeze(dim=2).repeat(1, 1, h*w)

            lam = ((x_-y_).pow(2) * ms * mc).sum(dim=(1, 2)).sqrt().sum(dim=0) / b
            return lats, latc, lam
        else:
            lam = ((x_-y_).pow(2) * ms).sum(dim=(1, 2)).sqrt().sum(dim=0) / b
            return lats, lam


class CWD_Module_YOLOv5(nn.Module):

    """
        Channel-Wise Distillation Module for YOLOv5
        https://arxiv.org/pdf/2011.13256.pdf
    """
    def __init__(self, t_net, s_net, alpha=35.0, T=1.0):
        super(CWD_Module_YOLOv5, self).__init__()

        # Initialize basic parameters
        self.t_net = t_net
        self.s_net = s_net

        self.alpha = alpha
        self.T = T

        # Setup connector functions(adaptive layers)
        c1 = [nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),]
        c2 = [nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),]

        self.c1 = nn.Sequential(*c1)
        self.c2 = nn.Sequential(*c2)

        self.modules_list = nn.ModuleList([self.c1, self.c2])

        initialize_weights(self.modules_list.modules())

        # Assign headers for print
        self.headers = ('loss', 'cwl_g1', 'cwl_g2')

    def forward(self, x, res_s=None):
        inputs = x

        # Teacher network forward
        with torch.no_grad():
            res1_t = self.t_net.model[0:5](inputs)
            res2_t = self.t_net.model[5:7](res1_t)
        
        # Student network forward
        if res_s is None:
            res1_s = self.s_net.model[0:5](inputs)
            res2_s = self.s_net.model[5:7](res1_s)
        else:
            res1_s = res_s[0]
            res2_s = res_s[1]

        cwl_g1 = self.channel_wise_loss(res1_t.detach(), res1_s, self.c1) * self.alpha / 4
        cwl_g2 = self.channel_wise_loss(res2_t.detach(), res2_s, self.c2) * self.alpha / 2
        loss = cwl_g1 + cwl_g2

        return loss, torch.Tensor([loss, cwl_g1, cwl_g2]).cuda().detach()

    def channel_wise_loss(self, x, y, connector=None):
        # x represents teacher's feature, y represents student's feature
        b, c, _, _ = x.size()

        x = x.view(b, c, -1)
        y = y.view(b, c, -1) if connector is None else connector(y).view(b, c, -1)

        phi_x = F.softmax(x / self.T, dim=2)
        phi_y = F.softmax(y / self.T, dim=2)

        cwl = (torch.log(phi_x / phi_y) * phi_x / 1000).sum() / b
        return cwl


class AWD_Module_YOLOv5(nn.Module):

    """
        Attention-Weighted Feature Distillation Module for YOLOv5
    """
    def __init__(self, t_net, s_net, alpha=35.0, T=1.0):
        super(AWD_Module_YOLOv5, self).__init__()
        
        # Initialize basic parameters
        self.t_net = t_net
        self.s_net = s_net

        self.alpha = 35.0
        self.T = T

        # Setup connector functions(adaptive layers)
        c1 = [nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),]
        c2 = [nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),]

        self.c1 = nn.Sequential(*c1)
        self.c2 = nn.Sequential(*c2)

        self.modules_list = nn.ModuleList([self.c1, self.c2])

        initialize_weights(self.modules_list.modules())

        # Assign headers for print
        self.headers = ('loss', 'law_g1', 'law_g2')

    def forward(self, x, res_s=None):
        inputs = x
        
        # Teacher network forward
        with torch.no_grad():
            res1_t = self.t_net.model[0:5](inputs)
            res2_t = self.t_net.model[5:7](res1_t)

        # Student network forward
        if res_s is None:
            res1_s = self.s_net.model[0:5](inputs)
            res2_s = self.s_net.model[5:7](res1_s)
        else:
            res1_s = res_s[0]
            res2_s = res_s[1]

        law_g1 = self.aw_loss(res1_t.detach(), res1_s, self.c1) * self.alpha / 4
        law_g2 = self.aw_loss(res2_t.detach(), res2_s, self.c2) * self.alpha / 2 
        loss = law_g1 + law_g2

        return loss, torch.Tensor([loss, law_g1, law_g2]).cuda().detach()

    def at(self, x):
        # Spatial attention map
        return x.pow(2).mean(dim=1)

    def aw_loss(self, x, y, connector=None):
        # x represents teacher's feature, y represents student's feature
        b, c, h, w = x.size()

        atx = self.at(x).view(b, -1)
        ms = h * w * F.softmax(atx / self.T, dim=1)
        ms = ms.unsqueeze(dim=1).repeat(1, c, 1)

        x = x.view(b, c, -1)
        y = y.view(b, c, -1) if connector is None else connector(y).view(b, c, -1)

        phix = F.softmax(x, dim=2)
        phiy = F.softmax(y, dim=2)

        law = (torch.log(phix / phiy) * phix * ms / 1000).sum() / b
        return law


###
class AT_Module_YOLOv5(nn.Module):

    """
        Attention Transfer Module for YOLOv5
        https://arxiv.org/pdf/1612.03928.pdf
    """
    def __init__(self, t_net, s_net):
        super(AT_Module_YOLOv5, self).__init__()
        
        # Initialize basic parameters
        self.t_net = t_net
        self.s_net = s_net

        self.modules_list = None

        # Assign headers for print
        self.headers = ('loss', 'lat_g1', 'lat_g2')

    def forward(self, x, res_s=None):
        inputs = x
        
        # Teacher network forward
        with torch.no_grad():
            res1_t = self.t_net.model[0:5](inputs)
            res2_t = self.t_net.model[5:7](res1_t)

        # Student network forward
        if res_s is None:
            res1_s = self.s_net.model[0:5](inputs)
            res2_s = self.s_net.model[5:7](res1_s)
        else:
            res1_s = res_s[0]
            res2_s = res_s[1]

        lat_g1 = self.at_loss(res1_t.detach(), res1_s)
        lat_g2 = self.at_loss(res2_t.detach(), res2_s)
        loss = lat_g1+lat_g2

        return loss, torch.Tensor([loss, lat_g1, lat_g2]).cuda().detach()


    def at(self, x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

    def at_loss(self, x, y):
        return (self.at(x) - self.at(y)).pow(2).mean()


class PWD_Module_YOLOv5(nn.Module):

    """
        Pair-Wise Distillation Module for YOLOv5
        https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.pdf
    """
    def __init__(self, t_net, s_net, scale=None):
        super(PWD_Module_YOLOv5, self).__init__()

        # Initialize basic parameters
        assert scale is None or scale < 1.0 and scale > 0, "Invalid scale parameter. Value range (0, 1.0)"
        self.maxpool = None
        if scale is not None:
            kernel_size = int(1.0 / scale)
            self.maxpool = nn.MaxPool2d(kernel_size=(kernel_size, kernel_size), ceil_mode=True)

        self.criterion = self.sim_distance

        self.t_net = t_net
        self.s_net = s_net

        self.modules_list = None

        # Assign headers for print
        self.headers = ('loss', 'pwl_g1', 'pwl_g2')

    def forward(self, x, res_s=None):
        inputs = x

        # Teacher network forward
        with torch.no_grad():
            res1_t = self.t_net.model[0:5](inputs)
            res2_t = self.t_net.model[5:7](res1_t)
        
        # Student network forward
        if res_s is None:
            res1_s = self.s_net.model[0:5](inputs)
            res2_s = self.s_net.model[5:7](res1_s)
        else:
            res1_s = res_s[0]
            res2_s = res_s[1]

        pwl_g1 = self.pair_wise_loss(res1_t.detach(), res1_s)
        pwl_g2 = self.pair_wise_loss(res2_t.detach(), res2_s)
        loss = pwl_g1 + pwl_g2

        return loss, torch.Tensor([loss, pwl_g1, pwl_g2]).cuda().detach()

    def pair_wise_loss(self, res_t, res_s):
        if self.maxpool is not None:
            res_t = self.maxpool(res_t)
            res_s = self.maxpool(res_s)

        return self.criterion(res_t, res_s)

    def channel_L2(self, feat):
        return (((feat**2).sum(dim=1))**0.5).reshape(feat.shape[0], 1, feat.shape[2], feat.shape[3]) + 1e-8

    def similarity(self, feat):
        b, c, _, _ = feat.size()
        feat = feat.float()
        tmp = self.channel_L2(feat)
        feat = feat / tmp
        feat = feat.reshape(b, c, -1)
        return torch.einsum('icm,icn->imn', [feat, feat])

    def sim_distance(self, feat1, feat2):
        b, _, h, w = feat1.size()
        sim_err = ((self.similarity(feat1) - self.similarity(feat2))**2) / ((h*w)**2) / b
        sim_dis = sim_err.sum()
        return sim_dis


class Non_Local_Module(nn.Module):

    """
        Non-Local Module with Embedded Gaussian
        https://arxiv.org/pdf/1711.07971v3.pdf
    """
    def __init__(self, c1, c_inner=None, subsample=True):
        super(Non_Local_Module, self).__init__()
        self.c1 = c1
        if c_inner is None:
            self.c_inner = c1 // 2
        else:
            self.c_inner = c_inner

        self.g = nn.Conv2d(c1, c1//2, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(
            nn.Conv2d(c1//2, c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1)
        )

        self.theta = nn.Conv2d(c1, c1//2, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(c1, c1//2, kernel_size=1, stride=1, padding=0)

        if subsample:
            max_pool = nn.MaxPool2d(2)
            self.g = nn.Sequential(
                self.g,
                max_pool
            )
            self.phi = nn.Sequential(
                self.phi,
                max_pool
            )

    def forward(self, x):
        b, c, h, w = x.size()

        g_x = self.g(x).view(b, self.c_inner, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(b, self.c_inner, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(b, self.c_inner, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.c_inner, h, w)
        y = self.W(y)

        return y


class NLD_Module_YOLOv5(nn.Module):

    """
        Non-local Distillation Module for YOLOv5
        https://openreview.net/pdf/1e6969024e2fab8681c02ff62a2dbfc4feedcff4.pdf
    """
    def __init__(self, t_net, s_net):
        super(NLD_Module_YOLOv5, self).__init__()

        # Initialize basic parameters
        self.t_net = t_net
        self.s_net = s_net

        # Setup connector functions(adaptive layers)
        c1 = [nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(256)]
        c2 = [nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(512)]

        self.c1 = nn.Sequential(*c1)
        self.c2 = nn.Sequential(*c2)

        self.n1 = Non_Local_Module(256)
        self.n2 = Non_Local_Module(512)

        self.modules_list = nn.ModuleList([self.c1, self.c2, self.n1, self.n2])

        initialize_weights(self.modules_list.modules())

        # Assign headers for print
        self.headers = ('loss', 'nld_g1', 'nld_g2')

    def forward(self, x, res_s=None):
        inputs = x

        # Teacher network forward
        with torch.no_grad():
            res1_t = self.t_net.model[0:5](inputs)
            res2_t = self.t_net.model[5:7](res1_t)
        
        # Student network forward
        if res_s is None:
            res1_s = self.s_net.model[0:5](inputs)
            res2_s = self.s_net.model[5:7](res1_s)
        else:
            res1_s = res_s[0]
            res2_s = res_s[1]

        nld_g1 = self.nld_loss(res1_t.detach(), res1_s, self.n1, self.c1)
        nld_g2 = self.nld_loss(res2_t.detach(), res2_s, self.n2, self.c2)

        loss = nld_g1+nld_g2

        return loss, torch.Tensor([loss, nld_g1, nld_g2]).cuda().detach()

    def nld_loss(self, res_t, res_s, nl_module, connector=None):
        b, c, h, w = res_t.size()
        nl_t = nl_module(res_t)
        nl_s = nl_module(connector(res_s)) if connector is not None else nl_module(res_s)

        return (nl_t-nl_s).pow(2).mean()
