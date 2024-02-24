import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d


from scipy import linalg
import warnings
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from PIL import Image
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
def get_m1_s1(batch_size,test_loader,inception_v3):
    dl_length=test_loader.__len__()
    imgs_num = dl_length *  batch_size
    pred_arr = np.zeros((imgs_num,2048))
    for step, data in enumerate(test_loader, 0):
        start = step * batch_size
        end = start + batch_size
        imgs, _,_, _,_,_ = data
        imgs=np.array(imgs)
        imgs=torch.tensor(imgs)
        imgs=imgs.to(torch.device("cuda"))

        pred = inception_v3(imgs)[0]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    m1 = np.mean(pred_arr, axis=0)
    s1 = np.cov(pred_arr, rowvar=False)
    return m1,s1

import clip
class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess

        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0,
                                                                                                 2.0])] +  # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
                                             clip_preprocess.transforms[:2] +  # to match CLIP input scale assumptions
                                             clip_preprocess.transforms[4:])  # + skip convert PIL to tensor

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text, generated_images):
        text_features = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if __name__ == '__main__':
    batch_size = 1
    num_workers = 1
    from dataprocess import Bird_Dataset_DF_GAN
    from torch.utils.data import DataLoader
    from torch.nn.functional import adaptive_avg_pool2d
    import torchvision.transforms as transforms

    test_dataset = Bird_Dataset_DF_GAN('test')
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'shuffle': False,
        'drop_last': True,
    }
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    clipevaluate = CLIPEvaluator(device)
    norm = transforms.Compose([
        transforms.ToTensor(),
    ])
    imgs_num = 300
    sum_pro = 0
    for step, data in enumerate(test_loader, 0):
        _, attrs, _, _, _, txt = data
        txt = list(txt)
        for i in range(txt.__len__()):
            if step * batch_size + i > imgs_num:
                break
            path = './outputs/txt2img-samples/samples_10/' + str(step * batch_size + i) + '.png'
            image = Image.open(path).convert('RGB')
            image = norm(image)
            image=torch.unsqueeze(image,0)

            sim=clipevaluate.txt_to_img_similarity(txt[i].replace("*", ""), image)
            sum_pro+=sim
    print('clip score:', sum_pro / imgs_num)

    pred_arr = np.zeros((imgs_num, 2048))

    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
    ])

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_v3 = InceptionV3([block_idx]).to(device)

    m1, s1 = get_m1_s1(batch_size,test_loader,inception_v3)
    for i in range(imgs_num):
        path='./outputs/txt2img-samples/samples_10/'+str(i)+'.png'
        fake=Image.open(path).convert('RGB')
        fake=norm(fake)
        fake=torch.unsqueeze(fake,0).to(device)
        pred = inception_v3(fake)[0]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[i:i+1] = pred.cpu().data.numpy().reshape(batch_size, -1)
        if (i % 10 == 0):
            print('step', i)

    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    print('fid value:',fid_value)

