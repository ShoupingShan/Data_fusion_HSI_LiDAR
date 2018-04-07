
import pickle as pkl
import numpy as np
import spectral
import pandas as pd
import scipy.io as io
import os

DATA_PATH = os.path.join(os.getcwd(),"Data")
output_image = io.loadmat(os.path.join(DATA_PATH, 'Indian_pines_gt.mat'))['indian_pines_gt']
height = output_image.shape[0]
width = output_image.shape[1]
targets = []
for j in range(height):
    for i in range(width):
        if output_image[j][i]==0 :
            continue
        else :
            targets.append(output_image[j][i])

unq, unq_idx = np.unique(targets, return_inverse=True)
unq_cnt = np.bincount(unq_idx)
count_mat = []
for i in range(len(unq_cnt)):
    count_mat.append(unq_cnt[i])
print("Polulation of target pixels of different classes: ", count_mat)

validation_scores = {'5x5': 86.19, '11x11':85.19, '21x21':97.31, '31x31':98.19, '37x37':99.56}
CLASSES = 16

total = sum(validation_scores.values())
credibility = {}
for keys,value in validation_scores.items():
    credibility[keys]=value/total

f = open('Predictions.pkl','rb')
output_predictions = {}
for i in range(5):
    for keys, values in (pkl.load(f).iteritems()):
        score = validation_scores[keys]
        for a in range(len(values)):
            for b in range(len(values)):
                if isinstance(values[a][b],int):
                    values[a][b] = np.zeros((16))
        output_predictions[keys] = np.asarray(values)*credibility[keys]

final_matrix = sum(output_predictions.values())

predictions=[]
cnf_mat =[[0 for x in range(CLASSES)] for  y in range(CLASSES)]
for i in range(len(final_matrix)):
    temp=[]
    for j in range(len(final_matrix[i])):
        if np.count_nonzero(final_matrix[i][j]) == 0 :
            temp.append(0)
        else:
            tmp = np.argmax(final_matrix[i][j])
            temp.append(tmp+1)
            if tmp == output_image[i][j]-1:
                cnf_mat[tmp][tmp] = cnf_mat[tmp][tmp] + 1
            else :
                cnf_mat[tmp][output_image[i][j]-1] = cnf_mat[tmp][output_image[i][j]-1] + 1
    predictions.append(temp)
for i in range(CLASSES):
    for j in range(CLASSES):
        cnf_mat[i][j] = 100*(float(cnf_mat[i][j])/count_mat[j])

df = pd.DataFrame(cnf_mat)
print(df)
predictions = np.array(predictions)