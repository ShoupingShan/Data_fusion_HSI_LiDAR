# -*- coding: utf-8 -*-  
import scipy.io
import numpy as np
from random import shuffle
import random
import spectral
import scipy.ndimage
# from skimage.transform import rotate
import os
import parameter
import matplotlib
import matplotlib.pyplot as plt

'''
载入数据
'''
DATA_name='DATA_DSM.mat'
DATA_PATH = os.path.join(os.getcwd(),"Data")
input_mat = scipy.io.loadmat(os.path.join(DATA_PATH, DATA_name))['DSM_300']
input_mat=input_mat[:,:,np.newaxis]
target_mat = scipy.io.loadmat(os.path.join(DATA_PATH, DATA_name))['GT_300']
# input_mat = input_mat[:,:,0:200] #只取前200个波段
HEIGHT = input_mat.shape[0]  #图像的高度
WIDTH = input_mat.shape[1]   #图像的宽度
BAND = 1    #图像的维度
PATCH_SIZE = parameter.patch_size
TRAIN_PATCH,TRAIN_LABELS,TEST_PATCH,TEST_LABELS = [],[],[],[]
CLASSES = []
COUNT = parameter.COUNT          #每一类的patch数量
OUTPUT_CLASSES = parameter.OUTPUT_CLASSES
TEST_FRAC = parameter.TEST_FRAC     #测试数据的比例
print("patch为",PATCH_SIZE)
# plt.imshow(target_mat)
# plt.show()
gt=target_mat[23:276,23:276]
un=np.unique(gt)
print(un)
# plt.imshow(gt)
# plt.show()

'''
把数据归一化到0~1
x = (x-min(X))/(max(X))
'''
input_mat = input_mat.astype(float)
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)

'''
按照每一个光谱带进行均值化
'''
MEAN_ARRAY = np.ndarray(shape=(BAND,),dtype=float)
for i in range(BAND):
    MEAN_ARRAY[i] = np.mean(input_mat[:,:,i])


def Patch(height_index, width_index):
    """
    返回一个均值化后的patch, 左上角坐标为(height_index, width_index)

    Inputs:
    height_index - 图像patch的左上角坐标的行数
    width_index  - 图像patch的左上角坐标的列数

    Outputs:
    mean_normalized_patch - 均值化后的patch,大小为 (PATCH_SIZE, PATCH_SIZE)
    左上角坐标位于 (height_index, width_index)
    """
    transpose_array = np.transpose(input_mat, (2, 0, 1))
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patch = transpose_array[:, height_slice, width_slice]
    # plt.imshow(patch[20,:,:])
    # plt.show()
    mean_normalized_patch = []
    for i in range(patch.shape[0]):
        mean_normalized_patch.append(patch[i] - MEAN_ARRAY[i])

    return np.array(mean_normalized_patch)

for i in range(OUTPUT_CLASSES):
    CLASSES.append([])
for i in range(HEIGHT - PATCH_SIZE + 1):
    for j in range( WIDTH - PATCH_SIZE + 1):
        curr_inp = Patch(i,j)
        curr_tar = target_mat[i + int((PATCH_SIZE - 1)/2), j + int((PATCH_SIZE - 1)/2)]  #对应的类标位置
        if(curr_tar!=0):                                 #忽略中心像素为0,即不需要分类的像素点
            CLASSES[curr_tar-1].append(curr_inp)  #存储每个类别所对应的像素邻域patch

for c in range(OUTPUT_CLASSES):
    class_population = len(CLASSES[c])   #每一类中所有的像素数
    test_split_size = int(class_population * TEST_FRAC)

    patches_of_current_class = CLASSES[c]
    shuffle(patches_of_current_class)

    # 切分出训练数据和测试数据
    TRAIN_PATCH.append(patches_of_current_class[:-test_split_size])
    TEST_PATCH.extend(patches_of_current_class[-test_split_size:])
    TEST_LABELS.extend(np.full(test_split_size, c, dtype=int))

'''
对于每一类训练集中样本少于COUNT的进行过采样,生成新的样本
'''
for i in range(OUTPUT_CLASSES):
    if (len(TRAIN_PATCH[i]) < COUNT and len(TRAIN_PATCH[i])!=0):
        tmp = TRAIN_PATCH[i]
        print(len(TRAIN_PATCH[i]))    #调试接口
        print(" ")  # 调试接口

        for j in range(int(COUNT / len(TRAIN_PATCH[i]))):
            shuffle(TRAIN_PATCH[i])
            TRAIN_PATCH[i] = TRAIN_PATCH[i] + tmp  #每一次循环随机增加一倍样本库
    shuffle(TRAIN_PATCH[i])
    TRAIN_PATCH[i] = TRAIN_PATCH[i][:COUNT]    #每类只取前COUNT个patch

TRAIN_PATCH = np.asarray(TRAIN_PATCH)
TRAIN_PATCH = TRAIN_PATCH.reshape((-1,BAND,PATCH_SIZE,PATCH_SIZE))

TRAIN_LABELS = np.array([])
for l in range(OUTPUT_CLASSES):
    TRAIN_LABELS = np.append(TRAIN_LABELS, np.full(COUNT, l, dtype=int))

'''
分组保存训练数据patch,每组一共2*COUNT个patch
'''
for i in range(int(len(TRAIN_PATCH)/(COUNT*2))):
    train_dict = {}
    start = i * (COUNT*2)
    end = (i+1) * (COUNT*2)
    file_name = 'Train_'+str(PATCH_SIZE)+'_'+str(i+1)+'.mat'
    train_dict["train_patch"] = TRAIN_PATCH[start:end]
    train_dict["train_labels"] = TRAIN_LABELS[start:end]
    scipy.io.savemat(os.path.join(DATA_PATH, file_name),train_dict)

'''
分组保存测试数据patch,每组一共2*COUNT个patch
'''
for i in range(int(len(TEST_PATCH)/(COUNT*2))):
    test_dict = {}
    start = i * (COUNT*2)
    end = (i+1) * (COUNT*2)
    file_name = 'Test_'+str(PATCH_SIZE)+'_'+str(i+1)+'.mat'
    test_dict["test_patch"] = TEST_PATCH[start:end]
    test_dict["test_labels"] = TEST_LABELS[start:end]
    scipy.io.savemat(os.path.join(DATA_PATH, file_name),test_dict)
