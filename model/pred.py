import torch
from sklearn.metrics import classification_report

from data.dataset import HSI_Loader
from model import HybridSN,HybridSN_BN_Attention

def pred_net(net, device, dataset, batch_size=128):
    # 加载训练集

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    # net.load_state_dict(torch.load('best_model_net_BN_Attention.pth', map_location=device))  # 加载模型参数
    net.load_state_dict(torch.load('best_model_net.pth', map_location=device))  # 加载模型参数
    net.eval()
    accuracy = 0
    n = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for curve, label in train_loader:
            # 将数据拷贝到device中
            curve = curve.to(device=device)
            label = label.to(device=device)
            # print(label)
            # 使用网络参数，输出预测结果
            out = net(curve)
            # 计算loss
            loss = criterion(out, label)
            pred = torch.argmax(out, dim=1)
            print(f'loss:{loss}')
            print(f'label:{label}')
            print(f'pred；{pred}')
            accuracy += torch.eq(pred,label).sum()
            n += label.shape[0]
        acc = accuracy / n
        print(f'准确率为：{acc}')
        # plot(pred,acc,2)
        # 生成分类报告
        classification = classification_report(label.detach().cpu().numpy(),pred.detach().cpu().numpy(), digits=4)
        print(classification)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = HSI_Loader('../data/sim_data/X_test.npy',
                               '../data/sim_data/y_test.npy')
    net = HybridSN()
    # net  =HybridSN_BN_Attention()
    # 将网络拷贝到deivce中
    net.to(device=device)
    pred_net(net,device,test_data)