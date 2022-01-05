import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from MoCoPnet import Net
from dataset import *
import matplotlib.pyplot as plt
from evaluation import psnr2, ssim
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="PyTorch D3Dnet")
parser.add_argument("--datasets", type=str, default=['SAITD','Hui','Anti-UAV'], help="Test datasets")
parser.add_argument("--save_img", type=bool, default=False, help="save super-resolved images")
parser.add_argument("--save_dir", type=str, default='./results/', help="save path")
parser.add_argument("--ckpt", default='./log/MoCoPnet.pth.tar', type=str, help="checkpoint path")
parser.add_argument("--scale_factor", type=int, default=4, help="scale factor")
parser.add_argument("--input_num", type=int, default=7, help="input frame number")
parser.add_argument("--test_dataset_dir", default='./data/test/', type=str, help="train_dataset")
parser.add_argument("--batch_size", type=int, default=1, help="Test batch size")
parser.add_argument("--device", type=int, default=0, help="GPU id")

global opt
opt = parser.parse_args()

def demo_test(net, test_loader, video_name):
    PSNR_list = []
    SSIM_list = []
    with torch.no_grad():
        for idx_iter, (LR, HR) in enumerate(test_loader):
            LR, HR = Variable(LR).to(opt.device), Variable(HR).to(opt.device)
            SR = net(LR)
            SR = torch.clamp(SR, 0, 1)
            
            # save images
            if not os.path.exists(opt.save_dir + opt.dataset + '/' + video_name + '/'):
                os.makedirs(opt.save_dir + opt.dataset + '/' + video_name + '/')
            SR_img = transforms.ToPILImage()(SR[0, :, :, :].cpu())
            SR_img.save(opt.save_dir + opt.dataset + '/' + video_name + '/' + str(idx_iter+1) + '.png')
            
            PSNR_list.append(psnr2(SR, HR[:, opt.input_num//2, :, :, :]))
            SSIM_list.append(ssim(SR, HR[:, opt.input_num//2, :, :, :]))

        PSNR_mean = float(torch.cat(PSNR_list, 0)[2:-2].data.cpu().mean())
        SSIM_mean = float(torch.cat(SSIM_list, 0)[2:-2].data.cpu().mean())
        print(opt.dataset + '\t' + video_name + ' psnr: '+ str(PSNR_mean) + ' ssim: ' + str(SSIM_mean))
        return PSNR_mean, SSIM_mean

def valid(net, dataset):
    
    PSNR_dataset = []
    SSIM_dataset = []

    video_list = os.listdir(opt.test_dataset_dir + dataset)
    for i in range(0, len(video_list)):
        video_name = video_list[i]
        test_set = TestSetLoader(opt.test_dataset_dir + dataset + '/' + video_name,
                                 scale_factor=opt.scale_factor, input_num=7)
        test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
        psnr, ssim = demo_test(net, test_loader, video_name)
        PSNR_dataset.append(psnr)
        SSIM_dataset.append(ssim)
    print(opt.dataset + '\t' + 'Avg' + ' psnr: '+ str(float(np.array(PSNR_dataset).mean())) + ' ssim: ' + str(float(np.array(SSIM_dataset).mean())))

def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path,filename))

def main():
    net = Net(opt.scale_factor)

    ckpt = torch.load(opt.ckpt, map_location='cuda:0')
    net.load_state_dict(ckpt['state_dict'])
        
    net = net.to(opt.device)
    for i in range(len(opt.datasets)):
        opt.dataset = opt.datasets[i]
        valid(net, opt.dataset)

if __name__ == '__main__':
    main()

