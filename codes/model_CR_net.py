import torch
import torch.nn as nn

from model_base import *

from net_CR_RDN import *

from metrics import *

from torch.optim import lr_scheduler

class ModelCRNet(ModelBase):
    def __init__(self, opts):
        super(ModelCRNet, self).__init__()
        self.opts = opts
        
        # create network
        self.net_G = RDN_residual_CR(self.opts.crop_size).cuda()
        self.print_networks(self.net_G)

        # load pre-trained model
        if self.opts.load_pretrained_model:
            checkpoint = torch.load(self.opts.pretrained_model)
            self.net_G.load_state_dict(checkpoint['network'])
        
        # Parallel training
        if len(self.opts.gpu_ids) > 1:
            print("Parallel training!")
            self.net_G = nn.DataParallel(self.net_G)

        # initialize optimizers
        if self.opts.optimizer == 'SGD':
            self.optimizer_G = torch.optim.SGD(self.net_G.parameters(), lr=opts.lr, momentum=0.9)
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.1)
        elif self.opts.optimizer == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opts.lr)
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer_G, step_size=self.opts.lr_step, gamma=0.5)
            
        self.loss_fn=nn.L1Loss()
                        
    def set_input(self, _input):
        inputs = _input
        self.cloudy_data = inputs['cloudy_data'].cuda()
        self.cloudfree_data = inputs['cloudfree_data'].cuda()
        self.SAR_data = inputs['SAR_data'].cuda()
        
    def forward(self):
        pred_CloudFree_data = self.net_G(self.cloudy_data, self.SAR_data)
        return pred_CloudFree_data

    def optimize_parameters(self):
                
        self.pred_Cloudfree_data = self.forward()

        self.loss_G = self.loss_fn(self.pred_Cloudfree_data, self.cloudfree_data)

        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step()  

        return self.loss_G.item()

    def get_current_scalars(self):
        losses = {}
        losses['PSNR_train']=PSNR(self.pred_Cloudfree_data.data, self.cloudfree_data)
        return losses

    def save_checkpoint(self, epoch):
        self.save_network(self.net_G, self.optimizer_G, epoch, self.lr_scheduler, self.opts.save_model_dir)
