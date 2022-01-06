import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from MoCoPnet import Net
from dataset import *
import matplotlib.pyplot as plt
from evaluation import psnr
from evaluation import psnr2, ssim
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="PyTorch MoCoPnet")
parser.add_argument("--save", default='./log', type=str, help="Save path")
parser.add_argument("--resume", default="", type=str, help="Resume path (default: none)")
parser.add_argument("--scale_factor", type=int, default=4, help="scale")
parser.add_argument("--input_num", type=int, default=7, help="input frame number")
parser.add_argument("--train_dataset_dir", default='./data/train/SAITD', type=str, help="train_dataset")
parser.add_argument("--val_dataset_dir", default='./data/test/SAITD', type=str, help="train_dataset")
parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--n_iters', type=int, default=100000, help='number of iterations to train')
parser.add_argument("--device", default=0, type=int, help="GPU id (default: 0)")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate. Default=4e-4")
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument("--milestones", type=int, default=[10000,20000,60000], help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=6")
parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loader to use, Default: 1")

global opt
opt = parser.parse_args()

def train(train_loader, scale_factor):
    loss_list = []
    psnr_list = []
    loss_epoch = []
    psnr_epoch = []
    
    net = Net(scale_factor)
    if opt.resume:
        ckpt = torch.load(opt.resume, map_location='cuda:0')
        net.load_state_dict(ckpt['state_dict'])
        iter_resume = ckpt['iter']
        loss_list = ckpt['loss']
        psnr_list = ckpt['psnr']
        
    net = net.cuda()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    criterion_MSE = torch.nn.MSELoss().cuda()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    for idx_iter, (LR, HR) in tqdm(enumerate(train_loader)):
        if opt.resume:
            idx_iter = iter_resume + idx_iter
        LR, HR = Variable(LR).cuda(), Variable(HR).cuda()
        SR = net(LR)

        loss = criterion_MSE(SR, HR[:, opt.input_num//2, :, :, :])
        loss_epoch.append(loss.detach().cpu())
        psnr_epoch.append(psnr(SR, HR[:, opt.input_num//2, :, :, :]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (idx_iter+1) % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            print(time.ctime()[4:-5] + ' iter---%d, loss_epoch---%f, PSNR---%f' % (idx_iter + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))
            save_checkpoint({
                'iter': idx_iter + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,
                'psnr': psnr_list,
            }, save_path=opt.save, filename='model' + str(scale_factor) + '_iter' + str(idx_iter+1) + '.pth.tar')
            loss_epoch = []
            psnr_epoch = []
            valid(net)

def demo_test(net, test_loader, video_name):
    PSNR_list = []
    SSIM_list = []
    with torch.no_grad():
        for idx_iter, (LR, HR) in enumerate(test_loader):
            LR, HR = Variable(LR).to(opt.device), Variable(HR).to(opt.device)
            SR = net(LR)
            SR = torch.clamp(SR, 0, 1)
            PSNR_list.append(psnr2(SR, HR[:, opt.input_num//2, :, :, :]))
            SSIM_list.append(ssim(SR, HR[:, opt.input_num//2, :, :, :]))

        PSNR_mean = float(torch.cat(PSNR_list, 0)[2:-2].data.cpu().mean())
        SSIM_mean = float(torch.cat(SSIM_list, 0)[2:-2].data.cpu().mean())
        print(video_name + ' psnr: '+ str(PSNR_mean) + ' ssim: ' + str(SSIM_mean))
        return PSNR_mean, SSIM_mean

def valid(net):
    
    PSNR_dataset = []
    SSIM_dataset = []

    video_list = os.listdir(opt.val_dataset_dir)
    for i in range(0, len(video_list)):
        video_name = video_list[i]
        test_set = TestSetLoader(opt.val_dataset_dir + '/' + video_name,
                                 scale_factor=opt.scale_factor, input_num=opt.input_num)
        test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
        psnr, ssim = demo_test(net, test_loader, video_name)
        PSNR_dataset.append(psnr)
        SSIM_dataset.append(ssim)
    print('Avg\t' + str(float(np.array(PSNR_dataset).mean())) + '\t' + str(float(np.array(SSIM_dataset).mean())))

def save_checkpoint(state, save_path, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path,filename))

def main():
    train_set = TrainSetLoader(opt)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
    train(train_loader, opt.scale_factor)

if __name__ == '__main__':
    main()

