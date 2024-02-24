import argparse, os, sys, datetime, glob, importlib, csv
import copy

import matplotlib.pyplot as plt
import numpy as np
import time
import torch

import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler
from bitsandbytes.optim import AdamW8bit
from accelerate import Accelerator
from ldm.util import instantiate_from_config
import random
from ldm.modules.diffusionmodules.util import make_ddim_timesteps
from tqdm import tqdm

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.modules.embedding_manager import EmbeddingManager

torch.backends.cudnn.benchmark = True
device = torch.device("cuda")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_model_path",type=str,default='./models/Stable-diffusion/v1-5-pruned-emaonly.safetensors',help="Path to pretrained model of stable diffusion.",)
    parser.add_argument("--batch_size",type=int,default=1,help="Batch size.",)
    parser.add_argument("--seed",type=int,default=42,help="the seed (for reproducible sampling)",)#噪声
    parser.add_argument("--model_path",type=str,default='C:/Users/TanTian/pythonproject/stable-diffusion-main/models/ldm/stable-diffusion-v1/v1-5-pruned2.ckpt',help="Model path.",)
    parser.add_argument("--config_path",type=str,default="C:/Users/TanTian/pythonproject/stable-diffusion-main/configs/stable-diffusion/v1-inference.yaml",help="Model path.",)
    # parser.add_argument("--model_path",type=str,default='C:/Users/TanTian/Downloads/textual_inversion-main/textual_inversion-main/models/ldm/text2img-large/model.ckpt',help="Model path.",)
    # parser.add_argument("--config_path",type=str,default="C:/Users/TanTian/Downloads/textual_inversion-main/textual_inversion-main/configs/latent-diffusion/txt2img-1p4B-finetune.yaml",help="Model path.",)
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

def train_one_step(args,imgs, tot_txt,masks,model,sampler):
    imgs=imgs.to(device)


    uc = model.get_learned_conditioning(args.batch_size * [""])

    tot_c = model.get_learned_conditioning([tot_txt[0]])

    timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=args.ddim_steps,
                                    num_ddpm_timesteps=model.num_timesteps, verbose=False)
    time_range = np.flip(timesteps)
    step=time_range[0]
    index = args.ddim_steps - 1
    ts = torch.full((args.batch_size,), step, device=device, dtype=torch.long)
    s = torch.zeros([args.batch_size]).to(device)
    img = torch.randn([args.batch_size, args.C, args.H // args.f, args.W // args.f], device=device)
    cond=model.get_learned_conditioning(tot_txt)
    out, _ = sampler.p_sample_ddim(img, cond, [cond, cond, cond], s, ts, index=index,
                                       use_original_steps=False,
                                       quantize_denoised=False, temperature=1.,
                                       noise_dropout=0., score_corrector=None,
                                       corrector_kwargs=None,
                                       unconditional_guidance_scale=args.scale,
                                       unconditional_conditioning=uc)

    img = out.detach()
    t = torch.randint(0, 1000, (args.batch_size,), device=device).long().to(device)

    model_output = model.apply_model(img, t, tot_c,[tot_c,tot_c,tot_c],tot_c)

    target = model.encode_first_stage(imgs)
    target = model.get_first_stage_encoding(target).detach()

    loss_simple = model.get_loss(model_output*masks, target*masks, mean=False).mean([1, 2, 3])
    model.logvar = model.logvar.to(device)
    logvar_t = model.logvar[t].to(device)
    loss = loss_simple / torch.exp(logvar_t) + logvar_t
    loss = model.l_simple_weight * loss.mean()
    loss_vlb = model.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
    loss_vlb = (model.lvlb_weights[t] * loss_vlb).mean()
    loss += (model.original_elbo_weight * loss_vlb)


    return loss



@torch.no_grad()
def sampling(args,cond,shape,uc,sampler,mask=None,x0=None):
    sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=0., verbose=False)
    timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=args.ddim_steps, num_ddpm_timesteps=sampler.model.num_timesteps,verbose=False)
    b = args.batch_size
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]
    C, H, W = shape
    shape = (b, C, H, W)
    if x0 is not None:
        img=x0
    else:
        img=torch.randn(shape).to(device)
    iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
    for i, step in enumerate(iterator):
        # if i<args.ddim_steps//2: #在训练阶段需要进行注释，在后期的生成阶段则取消注释进行跳过。
        #     continue

        index = total_steps - i - 1
        ts = torch.full((b,), step, device=device, dtype=torch.long)
        s = torch.zeros([b]).to(device)

        if mask is not None:
            img_orig = sampler.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
            img = img_orig * (1 - mask) + (mask) * img
            # img=img_orig

        img, _ = sampler.p_sample_ddim(img, cond, [], s, ts, index=index,
                            use_original_steps=False,
                            quantize_denoised=False, temperature=1.,
                            noise_dropout=0., score_corrector=None,
                            corrector_kwargs=None,
                            unconditional_guidance_scale=args.scale,
                            unconditional_conditioning=uc)
    return img


def get_f_mask(imgs,masks,model):
    imgs_masked = imgs * (1 - masks)-masks*torch.ones_like(imgs)
    encoder_imgs_masked = model.encode_first_stage(imgs_masked)
    encoder_imgs_masked = model.get_first_stage_encoding(encoder_imgs_masked).detach()
    imgs=imgs * (1 - masks)+masks*torch.ones_like(imgs)
    encoder_imgs = model.encode_first_stage(imgs)
    encoder_imgs = model.get_first_stage_encoding(encoder_imgs).detach()
    sim = 1-torch.nn.CosineSimilarity()(encoder_imgs, encoder_imgs_masked)
    masks = torch.unsqueeze(sim, dim=1).detach()
    return masks


def train():
    args = parse_args()
    seed_everything(args.seed)  # 设置噪声因子

    config = OmegaConf.load(args.config_path)  # 加载参数
    config = config.model

    model = LatentDiffusion(**config.get("params", dict()))

    checkpoint = torch.load('./models/ldm/stable-diffusion-v1/v1-5-pruned.ckpt')['state_dict']
    model.load_state_dict(checkpoint,strict=False)

    model.to(device)
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    embedding_params = list(model.embedding_manager.embedding_parameters())
    optimizer = AdamW8bit(embedding_params, lr=0.02)

    from ldm.data.personalized import PersonalizedBase
    train_dataset = PersonalizedBase(data_root='./train images/6/')
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'shuffle': True,
        'drop_last': True,
    }
    train_loader = DataLoader(train_dataset, **loader_kwargs)
    for param in model.cond_stage_model.parameters():
        param.requires_grad_(True)

    for param in model.model.parameters():
        param.requires_grad_(True)

    for param in model.embedding_manager.embedding_parameters():
        param.requires_grad_(True)

    for epoch in range(1000):
        for step,data in enumerate(train_loader, 0):
            txt,img,mask=data['caption'],data['image'],data['mask']
            img=torch.transpose(img,1,3)
            img=torch.transpose(img,2,3)
            img=img.to(device)
            mask=torch.unsqueeze(mask,1)
            mask=mask.to(device)
            loss = model.training_step({'caption': txt, 'jpg': img, 'masks': mask}, None)  # ddpm 344
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("saved",epoch)
        model.embedding_manager.save("./checkpoints/6/embeddings"+str(epoch)+".pt")

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        with torch.no_grad():
            sampler = DDIMSampler(model)
            sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=0., verbose=False)
            uc = sampler.model.get_learned_conditioning(args.batch_size * [''])
            shape = [args.C, args.H // args.f, args.W // args.f]
            cond = sampler.model.get_learned_conditioning(args.batch_size * ['a photo of a *'])
            samples_ddim = sampling(args, cond, shape, uc, sampler)
            x_samples_ddim = sampler.model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clip(x_samples_ddim, -1, 1)
            x_samples_ddim = x_samples_ddim.float()
            img = x_samples_ddim[0]
            img = torch.clip(img, -1, 1)
            img = (img + 1) / 2
            img = torch.transpose(img, 0, 2)
            img = torch.transpose(img, 0, 1)
            img = img.cpu().detach()
            plt.imsave('./samples/6/'+str(epoch)+'.png',np.array(img))
        model.train()
        for param in model.parameters():
            param.requires_grad = True


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
        for j in range(64):
            if (masks0[0][0][i][j] != 0):
                mn_dis = 100
                tarx = 0
                tary = 0
                for ii in range(64):
                    for jj in range(64):
                        if (masks1[0][0][ii][jj] != 0):
                            dis = np.abs(cntx0[i] - cntx1[ii]) + np.abs(cnty0[j] - cnty1[jj])
                            if (dis < mn_dis):
                                mn_dis = dis
                                tarx = ii
                                tary = jj
                encoder_imgs0[:, :, i, j] = encoder_imgs1[:, :, tarx, tary]

    return encoder_imgs0

def test():
    args = parse_args()
    seed_everything(args.seed)  # 设置噪声因子
    config = OmegaConf.load(args.config_path)  # 加载参数
    config = config.model

    model = LatentDiffusion(**config.get("params", dict()))

    # checkpoint = torch.load('./checkpoints/state_epoch_000.pth')['model']
    # model.load_state_dict(checkpoint,strict=False)

    checkpoint = torch.load('./models/ldm/stable-diffusion-v1/v1-5-pruned.ckpt')['state_dict']
    model.load_state_dict(checkpoint,strict=False)

    model.embedding_manager.load('./checkpoints/6/embeddings10.pt')
    model.eval()
    model.to(device)
    sampler = DDIMSampler(model)
    del checkpoint
    del model
    del config

    from dataprocess import Clothing_Dataset
    train_dataset = Clothing_Dataset()

    with torch.no_grad():
        data = train_dataset.__getitem__(694)
        imgs0, masks0, txts, _, txt0 = data
        print(txts)
        masks0 = masks0[2]
        masks0 = masks0.to(device)
        imgs0 = imgs0.to(device)
        imgs0 = torch.unsqueeze(imgs0, dim=0)
        masks0 = torch.unsqueeze(masks0, dim=0)
        encoder_imgs0 = sampler.model.encode_first_stage(imgs0)
        encoder_imgs0 = sampler.model.get_first_stage_encoding(encoder_imgs0).detach()
        masks0c = get_f_mask(imgs0, masks0, sampler.model)
        masks0b = get_f_mask(imgs0, 1 - masks0, sampler.model)
        masks_argmax = torch.cat([masks0c, masks0b], dim=1)
        masks_argmax = torch.nn.Softmax(dim=1)(masks_argmax)
        masks_argmax = torch.argmax(masks_argmax, dim=1)
        masks_argmax = torch.unsqueeze(masks_argmax, dim=0)
        ones = torch.ones_like(masks0c)
        zeros = torch.zeros_like(masks0c)
        masks0 = torch.where(masks_argmax == 0, ones, zeros)


        data = train_dataset.__getitem__(263)
        imgs1, masks1, txts, _, txt1 = data
        print(txts)
        masks1 = masks1[0]
        masks1 = masks1.to(device)
        imgs1 = imgs1.to(device)
        imgs1 = torch.unsqueeze(imgs1, dim=0)
        masks1 = torch.unsqueeze(masks1, dim=0)
        encoder_imgs1 = sampler.model.encode_first_stage(imgs1)
        encoder_imgs1 = sampler.model.get_first_stage_encoding(encoder_imgs1).detach()
        masks1c = get_f_mask(imgs1, masks1, sampler.model)
        masks1b = get_f_mask(imgs1, 1 - masks1, sampler.model)
        masks_argmax = torch.cat([masks1c, masks1b], dim=1)
        masks_argmax = torch.nn.Softmax(dim=1)(masks_argmax)
        masks_argmax = torch.argmax(masks_argmax, dim=1)
        masks_argmax = torch.unsqueeze(masks_argmax, dim=0)
        masks1 = torch.where(masks_argmax == 0, ones, zeros)

        encoder_imgs0 = clothing_mapping(masks0, masks1, copy.deepcopy(encoder_imgs0), encoder_imgs1)

        uc = sampler.model.get_learned_conditioning(args.batch_size * [''])
        shape = [args.C, args.H // args.f, args.W // args.f]

        cond = sampler.model.get_learned_conditioning(args.batch_size * ['*'])
        samples_ddim = sampling(args,cond,shape,uc,sampler,masks0,encoder_imgs0)
        #samples_ddim=encoder_imgs0
        x_samples_ddim = sampler.model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clip(x_samples_ddim, -1, 1)
        x_samples_ddim = x_samples_ddim.float()
        for i in range(args.batch_size):
            img = x_samples_ddim[i]
            img = torch.clip(img, -1, 1)
            img = (img + 1) / 2
            img = torch.transpose(img, 0, 2)
            img = torch.transpose(img, 0, 1)
            img = img.cpu().detach()
            plt.imshow(img)
            plt.imsave('./tmp.png',np.array(img))
            plt.show()


def save_clothing():
    args = parse_args()
    seed_everything(args.seed)  # 设置噪声因子
    config = OmegaConf.load(args.config_path)  # 加载参数
    config = config.model

    model = LatentDiffusion(**config.get("params", dict()))

    # checkpoint = torch.load('./checkpoints/state_epoch_000.pth')['model']
    # model.load_state_dict(checkpoint,strict=False)

    checkpoint = torch.load('./models/ldm/stable-diffusion-v1/v1-5-pruned.ckpt')['state_dict']
    model.load_state_dict(checkpoint, strict=False)

    model.embedding_manager.load('./checkpoints/0/embeddings105.pt')
    model.eval()
    model.to(device)
    sampler = DDIMSampler(model)
    del checkpoint
    del model
    del config

    from dataprocess import Clothing_Dataset
    train_dataset = Clothing_Dataset()

    with torch.no_grad():
        data = train_dataset.__getitem__(263)
        imgs0, masks0, txts, _, txt0 = data
        print(txts)
        masks0 = masks0[0]

        img=imgs0*masks0
        img=(1+img)/2
        img=torch.transpose(img,0,2)
        img=torch.transpose(img,0,1)
        plt.imsave('./train images/6/tmp.png',np.array(img))

        masks0 = masks0.to(device)
        imgs0 = imgs0.to(device)
        imgs0 = torch.unsqueeze(imgs0, dim=0)
        masks0 = torch.unsqueeze(masks0, dim=0)
        encoder_imgs0 = sampler.model.encode_first_stage(imgs0)
        encoder_imgs0 = sampler.model.get_first_stage_encoding(encoder_imgs0).detach()
        masks0c = get_f_mask(imgs0, masks0, sampler.model)
        masks0b = get_f_mask(imgs0, 1 - masks0, sampler.model)
        masks_argmax = torch.cat([masks0c, masks0b], dim=1)
        masks_argmax = torch.nn.Softmax(dim=1)(masks_argmax)
        masks_argmax = torch.argmax(masks_argmax, dim=1)
        masks_argmax = torch.unsqueeze(masks_argmax, dim=0)
        ones = torch.ones_like(masks0c)
        zeros = torch.zeros_like(masks0c)
        masks0 = torch.where(masks_argmax == 0, ones, zeros)

        print(masks0.shape)
        plt.imsave('./masks/6/tmp.png',np.array(masks0[0][0].cpu().detach()))


if __name__ == '__main__':
    train()
