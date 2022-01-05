from data_utils import *
import matplotlib.pyplot as plt
import imresize

class TrainSetLoader(Dataset):
    def __init__(self, cfg):
        super(TrainSetLoader).__init__()
        self.trainset_dir = cfg.train_dataset_dir
        self.scale = cfg.scale_factor
        self.patch_size = cfg.patch_size
        self.n_iters = cfg.n_iters * cfg.batch_size
        self.input_num = cfg.input_num
        self.video_list = os.listdir(cfg.train_dataset_dir)
        self.tranform = augumentation()
    def __getitem__(self, idx):
        idx_video = random.randint(0, len(self.video_list)-1)
        idx_frame = random.randint(self.input_num//2, len(os.listdir(self.trainset_dir + '/' + self.video_list[idx_video]))-self.input_num//2-1)
        HR = []
        LR = []
        for i in range(-self.input_num//2+1, self.input_num//2+1):
            img_hr = Image.open(self.trainset_dir + '/' + self.video_list[idx_video] + '/' + str(idx_frame+i) + '.png')
            img_hr = np.array(img_hr, dtype=np.float32)/255.0
            img_lr = imresize.imresize(img_hr, 1/self.scale, method='bicubic')
            img_hr = np.array(img_hr, dtype=np.float32)[np.newaxis,:]
            img_lr = np.array(img_lr, dtype=np.float32)[np.newaxis,:]
            HR.append(img_hr)
            LR.append(img_lr)

        HR = np.stack(HR, 0)
        LR = np.stack(LR, 0)

        HR, LR = random_crop(HR, LR, self.patch_size, self.scale)
        

        HR, LR = self.tranform(HR, LR)

        HR = torch.from_numpy(np.ascontiguousarray(HR.copy()))
        LR = torch.from_numpy(np.ascontiguousarray(LR.copy()))

        return LR, HR
    def __len__(self):
        return self.n_iters

class ValidSetLoader(Dataset):
    def __init__(self, cfg):
        super(ValidSetLoader).__init__()
        self.validset_dir = cfg.valid_dataset_dir
        self.scale = cfg.scale_factor
        self.patch_size = cfg.patch_size
        self.n_iters = cfg.n_iters * cfg.batch_size
        self.input_num = cfg.input_num
        self.video_list = os.listdir(cfg.valid_dataset_dir)
        self.tranform = augumentation()
    def __getitem__(self, idx):
        idx_video = random.randint(0, len(self.video_list)-1)
        idx_frame = random.randint(0, len(os.listdir(self.trainset_dir + '/' + self.video_list[idx_video]))-7)
        HR = []
        LR = []
        for i in range(-self.input_num//2+1, self.input_num//2+1):
            img_hr = Image.open(self.trainset_dir + '/' + self.video_list[idx_video] + '/' + str(idx_frame+i) + '.png')
            img_hr = np.array(img_hr, dtype=np.float32)/255.0
            img_lr = imresize.imresize(img_hr, 1/self.scale, method='bicubic')
            img_hr = np.array(img_hr, dtype=np.float32)[np.newaxis,:]
            img_lr = np.array(img_lr, dtype=np.float32)[np.newaxis,:]
            HR.append(img_hr)
            LR.append(img_lr)

        HR = np.stack(HR, 0)
        LR = np.stack(LR, 0)

        HR, LR = random_crop(HR, LR, self.patch_size, self.scale)
        

        HR, LR = self.tranform(HR, LR)

        HR = torch.from_numpy(np.ascontiguousarray(HR.copy()))
        LR = torch.from_numpy(np.ascontiguousarray(LR.copy()))

        return LR, HR
    def __len__(self):
        return self.n_iters


class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor, input_num):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.upscale_factor = scale_factor
        self.img_list = os.listdir(self.dataset_dir)
        self.totensor = transforms.ToTensor()
        self.input_num = input_num
    def __getitem__(self, idx):
        HR = []
        LR = []
        bicubic = []
        for idx_frame in range(idx - self.input_num//2, idx + self.input_num//2+1):
            if idx_frame < 0:
                idx_frame = 0
            if idx_frame > len(self.img_list) - 1:
                idx_frame = len(self.img_list) - 1
            try:
                img_HR = Image.open(self.dataset_dir + '/' + str(idx_frame) + '.png')
                img_HR = np.array(img_HR, dtype=np.float32)/255.0
            except:
                img_HR = Image.open(self.dataset_dir + '/' + str(idx_frame) + '.bmp')
                img_HR = np.array(img_HR, dtype=np.float32)/255.0
                img_HR = img_HR[:,:,0]
            # img_HR = np.array(img_HR, dtype=np.float32)/255.0
            img_LR = imresize.imresize(img_HR, 1/self.upscale_factor, method='bicubic')
            bicubic1 = imresize.imresize(img_LR, self.upscale_factor, method='bicubic')
            img_HR = np.array(img_HR, dtype=np.float32)[np.newaxis,:]
            img_LR = np.array(img_LR, dtype=np.float32)[np.newaxis,:]
            bicubic.append(np.array(bicubic1, dtype=np.float32)[np.newaxis,:])
            HR.append(img_HR)
            LR.append(img_LR)
            
        HR = np.stack(HR, 0)
        LR = np.stack(LR, 0)
        bicubic = np.stack(bicubic, 0)
        
        HR = torch.from_numpy(np.ascontiguousarray(HR.copy()))
        LR = torch.from_numpy(np.ascontiguousarray(LR.copy()))
        bicubic = torch.from_numpy(np.ascontiguousarray(bicubic.copy()))
        return LR, HR

    def __len__(self):
        return len(self.img_list)

class TestSetLoader_bic(Dataset):
    def __init__(self, dataset_dir, scale_factor):
        super(TestSetLoader_bic).__init__()
        self.dataset_dir = dataset_dir
        self.upscale_factor = scale_factor
        self.img_list = os.listdir(self.dataset_dir)
        self.totensor = transforms.ToTensor()
    def __getitem__(self, idx):
        HR = []
        LR = []
        bicubic = []
        for idx_frame in range(idx - 3, idx + 4):
            if idx_frame < 0:
                idx_frame = 0
            if idx_frame > len(self.img_list) - 1:
                idx_frame = len(self.img_list) - 1
            try:
                img_HR = Image.open(self.dataset_dir + '/' + str(idx_frame) + '.png')
                img_HR = np.array(img_HR, dtype=np.float32)/255.0
            except:
                img_HR = Image.open(self.dataset_dir + '/' + str(idx_frame) + '.bmp')
                img_HR = np.array(img_HR, dtype=np.float32)/255.0
                img_HR = img_HR[:,:,0]
            img_LR = imresize.imresize(img_HR, 1/self.upscale_factor, method='bicubic')
            bicubic1 = imresize.imresize(img_LR, self.upscale_factor, method='bicubic')
            img_HR = np.array(img_HR, dtype=np.float32)[np.newaxis,:]
            img_LR = np.array(img_LR, dtype=np.float32)[np.newaxis,:]
            bicubic.append(np.array(bicubic1, dtype=np.float32)[np.newaxis,:])
            HR.append(img_HR)
            LR.append(img_LR)
            
        HR = np.stack(HR, 0)
        LR = np.stack(LR, 0)
        bicubic = np.stack(bicubic, 0)
        
        HR = torch.from_numpy(np.ascontiguousarray(HR.copy()))
        LR = torch.from_numpy(np.ascontiguousarray(LR.copy()))
        bicubic = torch.from_numpy(np.ascontiguousarray(bicubic.copy()))
        return LR, HR, bicubic

    def __len__(self):
        return len(self.img_list)

class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[:, :, :, ::-1]
            target = target[:, :, :, ::-1]
        if random.random()<0.5:
            input = input[:, :, ::-1, :]
            target = target[:, :, ::-1, :]
        if random.random()<0.5:
            input = input.transpose(0, 1, 3, 2)#C N H W
            target = target.transpose(0, 1, 3, 2)
        return input, target

def random_crop(HR, LR, patch_size_lr, scale):
    b, _, h_hr, w_hr = HR.shape
    h_lr = h_hr // scale
    w_lr = w_hr // scale
    if h_lr <= patch_size_lr:
        idx_h = 1
        idx_w = 1
    else:
        idx_h = random.randint(10, h_lr - patch_size_lr - 10)
        idx_w = random.randint(10, w_lr - patch_size_lr - 10)

    h_start_hr = (idx_h - 1) * scale
    h_end_hr = (idx_h - 1 + patch_size_lr) * scale
    w_start_hr = (idx_w - 1) * scale
    w_end_hr = (idx_w - 1 + patch_size_lr) * scale

    h_start_lr = idx_h - 1
    h_end_lr = idx_h - 1 + patch_size_lr
    w_start_lr = idx_w - 1
    w_end_lr = idx_w - 1 + patch_size_lr

    HR0 = HR[:, :, h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    LR0 = LR[:, :, h_start_lr:h_end_lr, w_start_lr:w_end_lr]
    return HR0, LR0