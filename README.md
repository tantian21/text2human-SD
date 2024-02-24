# 基于扩散模型Stable DDiffusion的人像图像生成

本工作使用的扩散模型为Stable Diffusion，其代码可以在[Diffusion](https://github.com/CompVis/stable-diffusion)中下载，对应的预训练模型使用v1-5-pruned.ckpt版本，其可以在[pretrained model](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt)下载。

# 数据集

Clothing Co Parsing(CCP)数据集是一个用于衣物分割和服装类别标注的数据集，数据集中共包含了2,098张图像。其中，1,004张图像有像素级注释，这意味着每个图像都有一个相同大小的分割掩码，分割掩码中每个像素位置都被标记出对应位置的对象属性，如“blouse”，“dress”，“sunglasses”和皮肤头发身体部位等信息。其余的1,094张图像具有图像级注释，即每张图像对应一个元组记录该图像中包含的所有服装类型的标签。

Clothing Co Parsing数据集中共有59种不同的标签，范围从0到58不等。“0”表示图像的背景，其余的标签代表衣着类别和身体部位。由于该数据集中缺少用于文本生成图像的描述句子。本文便通过将图像中的标签信息进行组合生成对应图像的描述句子。生成的描述句子结构形如“a person in a dress and a blouse”。其可在[相应网站](https://github.com/bearpaw/clothing-co-parsing)下载。

将数据集下载完成后保存到相应的./data路径下，dataprocess.py是数据集加载程序。

![结果图](/fig16.png)

# 文本生成人像

由于扩散模型本身就是面向的文本生成图像任务，因此只需要在CCP数据集上进行微调，模型就能很好的生成相应图像。

将下载好的预训练模型放在text-to-human/models/ldm/stable-diffusion-v1/路径下，在text-to-human/路径下使用train.py中的train()函数进行训练预训练模型，之后再使用test()函数进行测试。

# 掩码生成人像

我们使用Clothing Co Parsing中的像素级标注作为掩码指导图像的生成。使用上面文本生成人像工作中得到的预训练模型进行生成任务。调用/mask-to-human/train.py中的test进行测试能生成最终的结果。

# 基于文本反演的人像图像生成

![结果图](/fig17.png)
