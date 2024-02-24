import argparse

import PIL.Image
import matplotlib.pyplot as plt
import torch
import transformers.image_transforms
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

import importlib
from ldm.models.diffusion.ddim import DDIMSampler
from accelerate import Accelerator
from bitsandbytes.optim import AdamW8bit
import torchvision
from ldm.models.diffusion.ddpm import LatentDiffusion
import numpy as np
import yaml
from safetensors.torch import load_file,save_file
from safetensors import safe_open
from ldm.modules.diffusionmodules.util import make_ddim_timesteps,noise_like
from tqdm import tqdm
import multiprocessing
import threading
import math
import torch.nn as nn
torch.backends.cudnn.benchmark = True

device = torch.device("cuda")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_model_path",type=str,default='./models/Stable-diffusion/v1-5-pruned-emaonly.safetensors',help="Path to pretrained model of stable diffusion.",)
    parser.add_argument("--batch_size",type=int,default=1,help="Batch size.",)
    parser.add_argument("--seed",type=int,default=42,help="the seed (for reproducible sampling)",)#噪声
    parser.add_argument("--model_path",type=str,default='C:/Users/TanTian/pythonproject/stable-diffusion-main/models/ldm/stable-diffusion-v1/v1-5-pruned2.ckpt',help="Model path.",)#噪声
    parser.add_argument("--config_path",type=str,default="C:/Users/TanTian/pythonproject/stable-diffusion-main/configs/stable-diffusion/v1-inference.yaml",help="Model path.",) # 噪声
    parser.add_argument("--prompt",type=str,default="a girl has long hair and in a black dress",help="Model path.",)#噪声
    parser.add_argument("--H",type=int,default=512,help="image height, in pixel space",)#图像高度，像素空间
    parser.add_argument("--W",type=int,default=512,help="image width, in pixel space",)#宽度
    parser.add_argument("--C",type=int,default=4,help="latent channels",)#潜在通道
    parser.add_argument("--f",type=int,default=8,help="downsampling factor",)#下采样因子
    parser.add_argument("--ddim_steps",type=int,default=20,help="number of ddim sampling steps",)#ddim采样步骤数
    parser.add_argument("--scale",type=float,default=7.5,help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",)#无条件制导刻度：eps=eps（x，空）+刻度*（eps（x，秒）-eps（x，空））
    parser.add_argument("--ddim_eta",type=float,default=0.0,help="ddim eta (eta=0.0 corresponds to deterministic sampling",)#ddim eta（eta=0.0对应于确定性采样
    parser.add_argument("--num_workers",type=int,default=1,help="Num work.",)#ddim eta（eta=0.0对应于确定性采样ddim eta（eta=0.0对应于确定性采样
    parser.add_argument("--num_epoch", type=int, default=50,help="Num epoch.", )  # ddim eta（eta=0.0对应于确定性采样ddim eta（eta=0.0对应于确定性采样
    parser.add_argument("--lr", type=int, default=1e-5,help="Learning rate.", )
    args = parser.parse_args()
    return args

@torch.no_grad()
def sampling(args,conds,tot_cond,mask_background, shape,sampler,uc,masks=None,x0=None):
    sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=0., verbose=False)
    b = args.batch_size
    C, H, W = shape
    shape = (b, C, H, W)
    #if init_img==None:
    img = torch.randn(shape, device=device)
    #else:
    #    img=init_img
    timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=args.ddim_steps, num_ddpm_timesteps=sampler.model.num_timesteps,verbose=False)

    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]

    iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

    for i, step in enumerate(iterator):
        if i<args.ddim_steps//5:
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            s = torch.zeros([img.shape[0]]).to(device)
            new_img=x0*mask_background
            #new_img=torch.zeros_like(img)

            if mask_background is not None:
                img_orig = sampler.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * (mask_background) + (1-mask_background) * img

            for i in range(len(masks)):
                mask=masks[i]
                cond=conds[i]
                out,_ = sampler.p_sample_ddim(img, cond,[], s, ts, index=index,
                                      use_original_steps=False,
                                      quantize_denoised=False, temperature=1.,
                                      noise_dropout=0., score_corrector=None,
                                      corrector_kwargs=None,
                                      unconditional_guidance_scale=args.scale,
                                      unconditional_conditioning=uc)

                new_img +=mask* out
            img=new_img
        else:
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            s = torch.zeros([img.shape[0]]).to(device)

            if mask_background is not None:
                img_orig = sampler.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * (mask_background) + (1-mask_background) * img

            out, _ = sampler.p_sample_ddim(img, tot_cond, [], s, ts, index=index,
                                           use_original_steps=False,
                                           quantize_denoised=False, temperature=1.,
                                           noise_dropout=0., score_corrector=None,
                                           corrector_kwargs=None,
                                           unconditional_guidance_scale=args.scale,
                                           unconditional_conditioning=uc)
            #img = x0 * mask_background*0.1+out*(1-mask_background*0.1)

            img = x0 * mask_background+out*(1-mask_background)
            #img=out
    return img

def get_f_mask(imgs,masks,model):
    encoder_imgs_masked = model.encode_first_stage(imgs - 2*(1-masks))
    encoder_imgs_masked = model.get_first_stage_encoding(encoder_imgs_masked).detach()
    #imgs=imgs * (1 - masks)+masks*torch.ones_like(imgs)
    encoder_imgs = model.encode_first_stage(imgs)
    encoder_imgs = model.get_first_stage_encoding(encoder_imgs).detach()
    sim = torch.nn.CosineSimilarity()(encoder_imgs, encoder_imgs_masked)
    masks = torch.unsqueeze(sim, dim=1).detach()

    return masks

class myThread ():   #继承父类threading.Thread
    def __init__(self, i,j,masks1,cntx0,cnty0,cntx1,cnty1,encoder_imgs0,encoder_imgs1):
        self.i=i
        self.j=j
        self.masks1=masks1
        self.cntx0=cntx0
        self.cnty0=cnty0
        self.cntx1=cntx1
        self.cnty1=cnty1
        self.encoder_imgs0=encoder_imgs0
        self.encoder_imgs1=encoder_imgs1
    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        tar_x = 0
        tar_y = 0
        dis = 100
        for ii in range(64):
            for jj in range(64):
                if (self.masks1[0][0][ii][jj] != 0):
                    if dis > ((self.cntx0[self.i] - self.cntx1[ii]) ** 2 + (self.cnty0[self.j] - self.cnty1[jj]) ** 2):
                        dis = ((self.cntx0[self.i] - self.cntx1[ii]) ** 2 + (self.cnty0[self.j] - self.cnty1[jj]) ** 2)
                        tar_x = ii
                        tar_y = jj
        self.encoder_imgs0[:, :, self.i, self.j] = self.encoder_imgs1[:, :, tar_x, tar_y]



def clothing_mapping(masks0,masks1,encoder_imgs0,encoder_imgs1):
    cntx0 = np.zeros([64])
    cnty0 = np.zeros([64])
    cntx1 = np.zeros([64])
    cnty1 = np.zeros([64])
    for i in range(64):
        for j in range(64):
            if (masks0[0][0][i][j] != 0):
                cntx0[i] += 1
                cnty0[j] += 1
            if (masks1[0][0][i][j] != 0):
                cntx1[i] += 1
                cnty1[j] += 1
    for i in range(1, 64):
        cntx0[i] += cntx0[i - 1]
        cnty0[i] += cnty0[i - 1]
        cntx1[i] += cntx1[i - 1]
        cnty1[i] += cnty1[i - 1]
    for i in range(64):
        cntx0[i] /= (cntx0[63] + 1e-8)
        cnty0[i] /= (cnty0[63] + 1e-8)
        cntx1[i] /= (cntx1[63] + 1e-8)
        cnty1[i] /= (cnty1[63] + 1e-8)
    for i in range(64):
        threads=[]
        for j in range(64):
            if (masks0[0][0][i][j] != 0):
                t=myThread(i,j,masks1,cntx0,cnty0,cntx1,cnty1,encoder_imgs0,encoder_imgs1)
                t.start()
                threads.append(t)
        for t in threads:
            t.join()

    return encoder_imgs0

def switch_clothing():
    args = parse_args()
    seed_everything(args.seed)  # 设置噪声因子
    config = OmegaConf.load(args.config_path)  # 加载参数
    config = config.model
    model = LatentDiffusion(**config.get("params", dict()))

    checkpoint = torch.load('./checkpoints/switching_clothes.pth')['model']
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    sampler = DDIMSampler(model)
    del checkpoint
    del model
    del config

    from dataprocess import Clothing_Dataset
    from torch.utils.data import DataLoader
    train_dataset = Clothing_Dataset()
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': False,
        'drop_last': True,
    }

    with torch.no_grad():
        data=train_dataset.__getitem__(4)
        imgs0, masks0, _, _, txt0 = data
        masks0=masks0[-1]
        masks0 = masks0.to(device)
        imgs0 = imgs0.to(device)
        imgs0=torch.unsqueeze(imgs0,dim=0)
        masks0=torch.unsqueeze(masks0,dim=0)

        encoder_imgs0 = sampler.model.encode_first_stage(imgs0)
        encoder_imgs0 = sampler.model.get_first_stage_encoding(encoder_imgs0).detach()

        masks0c = get_f_mask(imgs0,masks0,sampler.model)
        masks0b = get_f_mask(imgs0, 1-masks0, sampler.model)

        masks_argmax = torch.cat([masks0c,masks0b], dim=1)
        masks_argmax = torch.nn.Softmax(dim=1)(masks_argmax)
        masks_argmax = torch.argmax(masks_argmax, dim=1)
        masks_argmax = torch.unsqueeze(masks_argmax, dim=0)
        ones = torch.ones_like(masks0c)
        zeros = torch.zeros_like(masks0c)
        masks0 = torch.where(masks_argmax == 0, ones, zeros)


        data = train_dataset.__getitem__(5)
        imgs1, masks1, _, _, txt1 = data
        masks1=masks1[-1]
        masks1 = masks1.to(device)
        imgs1 = imgs1.to(device)
        imgs1=torch.unsqueeze(imgs1,dim=0)
        masks1=torch.unsqueeze(masks1,dim=0)

        encoder_imgs1 = sampler.model.encode_first_stage(imgs1)
        encoder_imgs1 = sampler.model.get_first_stage_encoding(encoder_imgs1).detach()


        masks1c = get_f_mask(imgs1,masks1,sampler.model)
        masks1b = get_f_mask(imgs1, 1-masks1, sampler.model)

        masks_argmax = torch.cat([masks1c,masks1b], dim=1)
        masks_argmax = torch.nn.Softmax(dim=1)(masks_argmax)
        masks_argmax = torch.argmax(masks_argmax, dim=1)
        masks_argmax = torch.unsqueeze(masks_argmax, dim=0)
        masks1 = torch.where(masks_argmax == 0, ones, zeros)


        encoder_imgs1=clothing_mapping(masks1,masks0,encoder_imgs1,encoder_imgs0)

        encoder_imgs0 = clothing_mapping(masks0, masks1, encoder_imgs0, encoder_imgs1)

        args.batch_size=1
        uc = sampler.model.get_learned_conditioning(args.batch_size * [""])
        c = sampler.model.get_learned_conditioning(txt0)
        shape = [args.C, args.H // args.f, args.W // args.f]
        samples_ddim = sampling(args, c, c, c, c,c, masks0,shape, sampler, uc,masks=masks0,x0=encoder_imgs0,init_img=encoder_imgs0,finetune=True)


        x_samples_ddim = sampler.model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clip(x_samples_ddim, -1, 1)
        x_samples_ddim = x_samples_ddim.float()
        img = x_samples_ddim[0]
        #torchvision.utils.save_image(img.data, './samples/tmp.png', nrow=4, range=(-1, 1), normalize=True)
        img = torch.clip(img, -1, 1)
        img = (img + 1) / 2
        img = torch.transpose(img, 0, 2)
        img = torch.transpose(img, 0, 1)
        img = img.cpu().detach()
        plt.imshow(img)
        plt.show()

def train_one_step(args,imgs, masks, txts, mask_background, tot_txt,model,sampler):
    imgs=imgs.to(device)
    imgs_background = imgs * mask_background.to(device)
    encoder_background = model.encode_first_stage(imgs_background)
    encoder_background = model.get_first_stage_encoding(encoder_background).detach()
    mask_background = get_f_mask(imgs, mask_background.to(device), model).detach()

    masks_argmax = mask_background

    f_masks = []
    c_s = []
    for i in range(txts.__len__()):
        mask = masks[i]  # [bt,1,512,512]
        mask = mask.to(device).detach()
        mask = get_f_mask(imgs, mask, model)
        c_s.append(model.get_learned_conditioning([txts[i][0]]).detach())
        masks_argmax = torch.cat([masks_argmax, mask], dim=1)
    masks_argmax = torch.nn.Softmax(dim=1)(masks_argmax)
    masks_argmax = torch.argmax(masks_argmax, dim=1)
    masks_argmax = torch.unsqueeze(masks_argmax, dim=0)
    ones = torch.ones_like(mask_background)
    zeros = torch.zeros_like(mask_background)
    mask_background = torch.where(masks_argmax == 0, ones, zeros).detach()

    for i in range(txts.__len__()):
        mask = torch.where(masks_argmax == i + 1, ones, zeros).detach()
        f_masks.append(mask)

    uc = model.get_learned_conditioning(args.batch_size * [""])

    tot_c = model.get_learned_conditioning([tot_txt[0]])

    timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=args.ddim_steps,
                                    num_ddpm_timesteps=model.num_timesteps, verbose=False)
    time_range = np.flip(timesteps)
    step=time_range[0]
    index = args.ddim_steps - 1
    ts = torch.full((args.batch_size,), step, device=device, dtype=torch.long)
    s = torch.zeros([args.batch_size]).to(device)
    new_img = encoder_background * mask_background
    img = torch.randn([args.batch_size, args.C, args.H // args.f, args.W // args.f], device=device)

    for i in range(len(masks)):
        mask = f_masks[i]
        cond = c_s[i]

        out, _ = sampler.p_sample_ddim(img, cond, cond, cond, cond, s, ts, index=index,
                                       use_original_steps=False,
                                       quantize_denoised=False, temperature=1.,
                                       noise_dropout=0., score_corrector=None,
                                       corrector_kwargs=None,
                                       unconditional_guidance_scale=args.scale,
                                       unconditional_conditioning=uc)


        new_img += mask * out
    img = new_img.detach()
    t = torch.randint(0, 1000, (args.batch_size,), device=device).long().to(device)

    model_output = model.apply_model(img, t, tot_c,tot_c,tot_c,tot_c,tot_c)

    target = model.encode_first_stage(imgs)
    target = model.get_first_stage_encoding(target).detach()

    loss_simple = model.get_loss(model_output, target, mean=False).mean([1, 2, 3])
    model.logvar = model.logvar.to(device)
    logvar_t = model.logvar[t].to(device)
    loss = loss_simple / torch.exp(logvar_t) + logvar_t
    loss = model.l_simple_weight * loss.mean()
    loss_vlb = model.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
    loss_vlb = (model.lvlb_weights[t] * loss_vlb).mean()
    loss += (model.original_elbo_weight * loss_vlb)


    return loss


def test():
    args=parse_args()
    seed_everything(args.seed)  # 设置噪声因子
    config = OmegaConf.load(args.config_path)  # 加载参数
    config=config.model
    model=LatentDiffusion(**config.get("params", dict()))
    #checkpoint={}
    #with safe_open('./models/ldm/stable-diffusion-v1/chomni.safetensors',framework="pt",device='cuda') as f:
    #    for k in f.keys():
    #        checkpoint[k]=f.get_tensor(k)
    #model.load_state_dict(checkpoint, strict=False)

    # checkpoint = torch.load('./models/ldm/stable-diffusion-v1/v1-5-pruned.ckpt')['state_dict']
    # model.load_state_dict(checkpoint,strict=False)

    checkpoint = torch.load('./checkpoints/state_epoch_004.pth')['model']
    model.load_state_dict(checkpoint,strict=False)
    model.to(device)
    model.eval()
    sampler = DDIMSampler(model)
    del checkpoint
    del model
    del config

    from dataprocess import Clothing_Dataset
    from torch.utils.data import DataLoader
    train_dataset = Clothing_Dataset()
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': False,
        'drop_last': True,
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    fs_imgs=torch.ones([1,3,512,512]).to(device)
    with torch.no_grad():
        for step, data in enumerate(train_loader, 0):
            imgs, masks, txts, mask_background, tot_txt = data
            imgs = imgs.to(device)
            imgs_background = imgs * mask_background.to(device)
            mask_background=get_f_mask(fs_imgs,mask_background.to(device),sampler.model)
            encoder_background = sampler.model.encode_first_stage(imgs_background)
            encoder_background = sampler.model.get_first_stage_encoding(encoder_background).detach()

            masks_argmax = mask_background

            f_masks=[]
            c_s=[]
            for i in range(txts.__len__()):
                mask=masks[i]#[bt,1,512,512]
                mask = mask.to(device)
                mask=get_f_mask(fs_imgs,mask,sampler.model)
                c_s.append(sampler.model.get_learned_conditioning([txts[i][0]]))
                masks_argmax=torch.cat([masks_argmax,mask],dim=1)
            masks_argmax=torch.nn.Softmax(dim=1)(masks_argmax)
            masks_argmax=torch.argmax(masks_argmax,dim=1)
            masks_argmax=torch.unsqueeze(masks_argmax,dim=0)
            ones=torch.ones_like(mask_background)
            zeros=torch.zeros_like(mask_background)
            mask_background=torch.where(masks_argmax==0,ones,zeros)

            for i in range(txts.__len__()):
                mask=torch.where(masks_argmax==i+1,ones,zeros)
                f_masks.append(mask)

            #f_masks.append(mask_background)
            #c_s.append(sampler.model.get_learned_conditioning('a girl in Times Square'))


            uc = sampler.model.get_learned_conditioning(args.batch_size * [""])
            shape = [args.C, args.H // args.f, args.W // args.f]

            tot_c=sampler.model.get_learned_conditioning('a girl in '+tot_txt[0]+' in Times Square')
            samples_ddim=sampling(args,c_s,tot_c, mask_background,shape,sampler,uc,masks=f_masks,x0=encoder_background)

            x_samples_ddim = sampler.model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clip(x_samples_ddim, -1, 1)
            x_samples_ddim = x_samples_ddim.float()

            img=x_samples_ddim[0]
            img=torch.clip(img,-1,1)
            img=(img+1)/2
            img=torch.transpose(img,0,2)
            img=torch.transpose(img,0,1)
            img=img.cpu().detach()

            plt.imsave('./tmp'+str(step)+'.png',np.array(img))

            plt.imshow(img)
            plt.show()
            # torchvision.utils.save_image(img.data, './samples/' + str(step) + '.png', nrow=4, range=(-1, 1), normalize=True)
            # break


def train():
    args = parse_args()
    seed_everything(args.seed)  # 设置噪声因子

    config = OmegaConf.load(args.config_path)  # 加载参数
    config = config.model
    model=LatentDiffusion(**config.get("params", dict()))

    sd = torch.load('./checkpoints/state_epoch_003.pth')['model']
    model.load_state_dict(sd,strict=False)
    del sd

    # checkpoint = torch.load('./models/ldm/stable-diffusion-v1/v1-5-pruned.ckpt')['state_dict']
    # model.load_state_dict(checkpoint,strict=False)
    # del checkpoint

    model.train()
    model.requires_grad_(True)
    model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=0., verbose=False)

    params_to_optimize = [{'params': model.model.parameters(), 'lr': 1e-05}]
    optimizer = AdamW8bit(params_to_optimize)


    from dataprocess import Clothing_Dataset
    from torch.utils.data import DataLoader
    train_dataset = Clothing_Dataset()
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': True,
        'drop_last': True,
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)



    accelerator = Accelerator(  # 实现多机多卡、单机多卡的分布式并行计算，另外还支持FP16半精度计算。
        gradient_accumulation_steps=1,  # 1
        mixed_precision='no',  # no
        cpu=False,  # False
    )
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_loader
    )

    for epoch in range(args.num_epoch):
        for step, data in enumerate(train_loader, 0):
            imgs, masks, txts, mask_background, tot_txt= data
            masks=1-mask_background
            txts=tot_txt
            masks=torch.clip(masks,0,1)
            imgs=imgs.to(device).requires_grad_()
            masks=masks.to(device)

            masksc = get_f_mask(imgs, masks, sampler.model)
            masksb = get_f_mask(imgs, 1 - masks, sampler.model)
            masks_argmax = torch.cat([masksc, masksb], dim=1)
            masks_argmax = torch.nn.Softmax(dim=1)(masks_argmax)
            masks_argmax = torch.argmax(masks_argmax, dim=1)
            masks_argmax = torch.unsqueeze(masks_argmax, dim=0)
            ones = torch.ones_like(masksc)
            zeros = torch.zeros_like(masksc)
            masks = torch.where(masks_argmax == 0, ones, zeros)

            txts=list(txts)

            loss_total = model.training_step({'caption':txts,'jpg':imgs,'masks':masks},step)#ddpm 344

            # loss_total=train_one_step(args,imgs, masks, txts, mask_background, tot_txt,model,sampler)
            accelerator.backward(loss_total)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if(step%100==0):
                print("step:",step,"loss:",loss_total)

            del loss_total
            del txts
            del imgs
            del masks




        if (epoch)%5 == 0:
            print('saved',epoch)
            state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
            torch.save(state, './checkpoints/state_epoch_%03d.pth' % (epoch%5+4))

if __name__ == '__main__':
    test()
