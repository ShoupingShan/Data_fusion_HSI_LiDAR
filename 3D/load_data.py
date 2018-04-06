import numpy as np
import torch
import scipy
import scipy.io as scio
from torch.autograd import Variable
from HSI_3d import CNN3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

CLASS_NUM = 16
TRAIN_PROB = 0.1 #训练数据所占的比例
IMAGE_SIZE = 13  #输入CNN网络的图片尺寸
# PCA 参数设置
N_COMPONENTS = 3   #降维之后的维数
'''
读取数据
'''

root = "./data/"
DATA = scio.loadmat(root+"Indian_pines.mat")
GT = scio.loadmat(root+"Indian_pines_gt.mat")
data=DATA['indian_pines']
gt=GT['indian_pines_gt']


# print(len(data))

# plt.imshow(gt)
# plt.show()

# subdata=data[:,:,100]
# plt.imshow(subdata)
# plt.show()




'''
PCA
'''



def de_reshape(x,H,W):
    '''
    还原成标准图像
    :param x: 二维矩阵,其高度为原始图像长*宽,宽度为光谱数 
    :param H: 原始图像高
    :param W: 原始图像宽
    :return:  原始三维矩阵
    '''
    s = x.shape
    rebuild = np.zeros([H,W,s[1]])
    for i in range(H):
        for j in range(W):
            rebuild[i,j,:] = x[i*H+j,:]
    return rebuild


size=data.shape
data_list = np.zeros([size[0]*size[1],size[2]])
pca = PCA(n_components=N_COMPONENTS,whiten=True)
for i in range(size[0]):
    for j in range(size[1]):
        data_list[i*size[0]+j,:] = data[i,j,:]

# plt.imshow(de_reshape(de_data,size[0],size[1])[:,:,0])
# plt.show()
de_data = de_reshape(pca.fit_transform(data_list),size[0],size[1] )  #进行PCA降维



'''
筛选数据,Indian pines一共16类
'''

##定义结构体用于存储各个类别的数据
class struct:
    def __init__(self):
        self.classes= 0  #类别编号
        self.list=[]     #存储像素原始位置,i*H+j
        self.size = 0    #存储某一类总数

class_data = []

for classes in range(CLASS_NUM+1):
    temp = struct()
    class_data.append(temp)
    class_data[classes].classes = classes
    count = 0
    for i in range(size[0]):
        for j in range(size[1]):
            if gt[i,j]==classes:
                class_data[classes].list.append(i*count+j)
                class_data[classes].size += 1
        count = count+1

print('数据初次分类整理完毕')

#每一类随机选取prob比例
def randomchoose(x,prob = TRAIN_PROB):
    '''
    随机筛选测试样本和训练数据
    :param x: 输入的结构体数组
    :param prob: 随机提取的百分比
    :return: slice表示训练数据,left表示测试数据
    '''
    slice = random.sample(x.list, int(x.size*prob))
    left = []
    for item in x.list:
        if item not in slice:
            left.append(item)
    return slice,left


train_data = []
test_data = []
for item in class_data:
    tra,tes = randomchoose(item)
    train_data.append(tra)
    test_data.append(tes)



'''
按像素对图像进行切割
'''
extend_map=[]
temp = np.zeros([size[0]+int(IMAGE_SIZE-1),size[1]+int(IMAGE_SIZE-1)])
for item in range(N_COMPONENTS):
    extend_map.append(temp)


for item in range(N_COMPONENTS):
     for i in range(size[0]):
      for j in range(size[1]):
        extend_map[item][int(i+IMAGE_SIZE/2),int(j+IMAGE_SIZE/2)] = de_data[i,j,item]
for i in range(size[0]):
    for j in range(size[1]):
        print(extend_map[1][int(i + IMAGE_SIZE / 2),int(j + IMAGE_SIZE / 2)])
        print( de_data[i, j, 1])

# extend_map[int(IMAGE_SIZE/2):int(size[0]+IMAGE_SIZE/2)-1,int(IMAGE_SIZE/2):int(size[1]+IMAGE_SIZE/2)-1,:] = de_data[:,:,:]
plt.figure(1)
plt.subplot(121)
plt.imshow(de_data[:,:,0].tolist())
plt.subplot(122)
plt.imshow(extend_map[0].tolist())
plt.show()







'''
转换数据为Tensor格式
'''
torch_data = torch.from_numpy(de_data)


# net = CNN3D()