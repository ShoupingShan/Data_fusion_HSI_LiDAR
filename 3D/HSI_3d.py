import torch
import torch.nn as nn


INPUT_CHANEL = 3


class CNN3D(nn.Module):
    ''''
    The 3Dcnn network
    '''
    def __init__(self):
        super(CNN3D, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(INPUT_CHANEL,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)),
            nn.Conv3d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2)),

        )
        self.fc_s = nn.Sequential(
            nn.Linear(128*3*3,1024),
            nn.ReLU(),
            nn.Dropout(0.5),

        )
        self.fc = nn.Linear(1024, 256)
        self.softmax = nn.Softmax()



    def forword(self,x):
        h = self.layers(x)
        h = self.fc_s(h)
        logits = self.fc(h)
        probs = self.softmax(logits)
        return probs

net = CNN3D()
#print(net)  # 打印网络