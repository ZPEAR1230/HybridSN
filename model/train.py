import torch
import torch.nn as nn
import torch
from torch import optim
import numpy as np
from data.dataset import HSI_Loader
from model import HybridSN,HybridSN_BN_Attention


def train(net,device,train_dataset,epochs=100,lr=0.0001,batch_size=128):

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    total_loss = 0
    best_loss = float('inf')
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 正向传播 +　反向传播 + 优化
            out= net(inputs)
            # pred = torch.argmax(out, dim=1)
            loss = criterion(out, labels)


            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model_net_BN_Attention.pth')

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (
        epoch + 1, total_loss / (epoch + 1), loss.item()))

    print('Finished Training')

if __name__ == '__main__':

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网络放到GPU上
    net = HybridSN_BN_Attention()
    net = net.to(device)
    train_dataset = HSI_Loader('../data/sim_data/X_train.npy',
                               '../data/sim_data/y_train.npy')
    train(net,device,train_dataset)



