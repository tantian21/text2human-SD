import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
data=scio.loadmat('C:/Users/TanTian/pythonproject/stable-diffusion-main/data/clothing-co-parsing-master/clothing-co-parsing-master/label_list.mat')
data=dict(data)['label_list'][0]
print(data)
for i in range(data.__len__()):
    print(data[i])