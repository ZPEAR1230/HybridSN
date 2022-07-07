import torch
from torch.utils.data import Dataset
import numpy as np
import random


class HSI_Loader(Dataset):

    def __init__(self,curve,label):
        Xtrain = np.load(curve)
        ytrain = np.load(label)
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len



if __name__ == "__main__":



    train_dataset = HSI_Loader('../data/sim_data/X_train.npy',
                         '../data/sim_data/y_train.npy')

    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               shuffle=False,
                                          )
    batch_size = 1024
    for pixel_curve, label in train_loader:
        # print(pixel_curve.reshape(batch_size, 1, -1).shape)
        print(pixel_curve.shape)
        print(label)
