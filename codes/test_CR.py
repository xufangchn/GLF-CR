import os
import torch
import argparse

from metrics import PSNR, SSIM

from dataloader import AlignedDataset, get_train_val_test_filelists

from net_CR_RDN import RDN_residual_CR

##########################################################
def test(CR_net, opts):

    _, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)
    
    data = AlignedDataset(opts, test_filelist)

    dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=opts.batch_sz,shuffle=False)

    iters = 0
    PSNR_13 = 0
    SSIM_13 = 0
    for inputs in dataloader:

        cloudy_data = inputs['cloudy_data'].cuda()
        cloudfree_data = inputs['cloudfree_data'].cuda()
        SAR_data = inputs['SAR_data'].cuda()
        file_name = inputs['file_name'][0]

        pred_cloudfree_data = CR_net(cloudy_data, SAR_data)
       
        psnr_13 = PSNR(pred_cloudfree_data, cloudfree_data)

        ssim_13 = SSIM(pred_cloudfree_data, cloudfree_data).item()

        print(iters, '  psnr_13:', format(psnr_13,'.4f'), '  ssim_13:', format(ssim_13,'.4f'))
    
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')

    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--input_data_folder', type=str, default='../data') 
    parser.add_argument('--data_list_filepath', type=str, default='../data/data.csv')

    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False) 
    parser.add_argument('--cloud_threshold', type=float, default=0.2) # only useful when is_use_cloudmask=True

    opts = parser.parse_args()

    CR_net = RDN_residual_CR(opts.crop_size).cuda()
    checkpoint = torch.load('../ckpt/CR_net.pth')
    CR_net.load_state_dict(checkpoint['network'])

    CR_net.eval()
    for _,param in CR_net.named_parameters():
        param.requires_grad = False

    test(CR_net, opts)

if __name__ == "__main__":
    main()
    