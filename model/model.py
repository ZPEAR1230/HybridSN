import torch
from torch import nn
from torch.utils.data import DataLoader
from parts import ChannelAttention,SpatialAttention
from data.dataset import HSI_Loader

class_num = 5

'''原始HybridSN'''
class HybridSN(nn.Module):
    def __init__(self, in_channels=1, out_channels=class_num):
        super(HybridSN, self).__init__()
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(7, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU()
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=32 * 18, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 17 * 17, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 16)
        )

    def forward(self, x):
        x = self.conv3d_features(x)
        x = x.view(x.size()[0], x.size()[1] * x.size()[2], x.size()[3], x.size()[4])
        x = self.conv2d_features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
'''带有Batch Normalization 和注意力机制（通道注意力和空间注意力）的HybridSN'''


class HybridSN_BN_Attention(nn.Module):
    def __init__(self, in_channels=1, out_channels=class_num):
        super(HybridSN_BN_Attention, self).__init__()
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(7, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.ca = ChannelAttention(32 * 18)
        self.sa = SpatialAttention()

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=32 * 18, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 17 * 17, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.conv3d_features(x)
        x = x.view(x.size()[0], x.size()[1] * x.size()[2], x.size()[3], x.size()[4])

        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.conv2d_features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 随机输入

    train_dataset = HSI_Loader('../data/sim_data/X_train.npy',
                               '../data/sim_data/y_train.npy')
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=128,
                                               shuffle=False)

    net = HybridSN_BN_Attention()
    net.to(device)
    for x,y in train_loader:
        print(x.shape)

        x = x.to(device)
        #
        #
        out = net(x)
        # pred = torch.argmax(out, dim=1)
        # print(y)
        # print(pred)




