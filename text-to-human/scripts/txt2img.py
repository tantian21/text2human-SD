import argparse, os, sys, glob
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


from scipy import linalg
import warnings
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class InceptionV3(nn.Module):
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }
    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        inception = models.inception_v3(pretrained=True)
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad
    def forward(self, inp):
        outp = []
        x = inp
        if self.resize_input:
            x = F.upsample(x, size=(299, 299), mode='bilinear', align_corners=True)
        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(#输入的文本
        "--prompt",
        type=str,
        nargs="?",
        default="a panda is eating a hamburger",
        help="the prompt to render"
    )
    parser.add_argument(#生成图像输出地址
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(#不保存网格，只保存单个样本。在评估大量样本时很有用
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(#不要保存单个样本。用于速度测量。
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(#ddim采样步骤数
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(#使用plms采样
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(#使用dpm_solver采样
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(#使用LAION400M模型
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(#如果启用，则在样本中使用相同的起始代码
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(#ddim eta（eta=0.0对应于确定性采样
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(#生成几个
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(#图像高度，像素空间
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(#宽度
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(#潜在通道
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(#下采样因子
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(#每个给定提示要生成多少个样本。A.k.A.批量
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(#网格中的行（默认值：n_samples）
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(#无条件制导刻度：eps=eps（x，空）+刻度*（eps（x，秒）-eps（x，空））
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(#参数文件
        "--config",
        type=str,
        default="C:/Users/TanTian/pythonproject/stable-diffusion-main/configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(#模型文件
        "--ckpt",
        type=str,
        default="C:/Users/TanTian/pythonproject/stable-diffusion-main/models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(#噪声
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)#设置噪声因子

    config = OmegaConf.load(f"{opt.config}")#加载参数
    model = load_model_from_config(config, f"{opt.ckpt}")#加载模型

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)#this line

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext

    batch_size=3
    num_workers=1
    from dataprocess import Bird_Dataset_DF_GAN
    from torch.utils.data import DataLoader
    import torchvision
    test_dataset = Bird_Dataset_DF_GAN('test')
    loader_kwargs = {
        'batch_size':batch_size,
        'num_workers': num_workers,
        'shuffle': False,
        'drop_last': True,
    }
    test_loader = DataLoader(test_dataset, **loader_kwargs)


    for step, data in enumerate(test_loader, 0):
        _, attrs, _, _, _,txt = data

        txt=list(txt)

        attrs0=[]
        for attr in attrs:
            tmp=test_dataset.ixtoword[int(attr[0][0].cpu().detach())]
            for i in range(1,5):
                if(int(attr[0][i].cpu().detach())!=0):
                    tmp=tmp+" " + test_dataset.ixtoword[int(attr[0][i].cpu().detach())]
            attrs0.append(tmp)
        attrs1 = []
        for attr in attrs:
            tmp = test_dataset.ixtoword[int(attr[1][0].cpu().detach())]
            for i in range(1, 5):
                if (int(attr[1][i].cpu().detach()) != 0):
                    tmp = tmp + " " + test_dataset.ixtoword[int(attr[1][i].cpu().detach())]
            attrs1.append(tmp)
        attrs2 = []
        for attr in attrs:
            tmp = test_dataset.ixtoword[int(attr[2][0].cpu().detach())]
            for i in range(1, 5):
                if (int(attr[2][i].cpu().detach()) != 0):
                    tmp = tmp + " " + test_dataset.ixtoword[int(attr[2][i].cpu().detach())]
            attrs2.append(tmp)
        txt=['a girl has long hair and in a black dress','a girl has long hair and in a black dress','a girl has long hair and in a black dress']
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    c = model.get_learned_conditioning(txt)
                    a0 = model.get_learned_conditioning(attrs0)
                    a1 = model.get_learned_conditioning(attrs1)
                    a2 = model.get_learned_conditioning(attrs2)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    a0=a0,
                                                    a1=a1,
                                                    a2=a2,
                                                    batch_size=batch_size,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta,
                                                    x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim=torch.clip(x_samples_ddim,-1,1)
                    x_samples_ddim=x_samples_ddim.float()
                    for i in range(batch_size):
                        torchvision.utils.save_image(x_samples_ddim[i].data, './outputs/txt2img-samples/samples/' + str(step*batch_size+i) + '.png', nrow=4,range=(-1, 1), normalize=True)

                    # img=x_samples_ddim[0].detach()
                    # img=torch.clip(img,-1,1)
                    # img=(img+1)/2
                    # img=(img*255)
                    # img=torch.transpose(img,0,2)
                    # img=np.array(img.cpu())
                    # x0=np.abs(img[:-1,:,:]-img[1:,:,:])
                    # x1=np.abs(img[:,:-1,:]-img[:,1:,:])
                    # x0=np.reshape(x0,[-1])
                    # x1=np.reshape(x1,[-1])
                    # x=np.concatenate((x0,x1),0)
                    # x=np.floor(x)
                    # unique, count = np.unique(x, return_counts=True)
                    # count=np.log(count)
                    # unique=np.array(unique,dtype=int)
                    # var=np.var(count[:-1]-count[1:])
                    # print(var)
                    # plt.bar(unique,count)
                    # plt.show()

        if step%10==0:
            print(str(step)+'/'+str(test_loader.__len__()))


if __name__ == "__main__":
    main()
