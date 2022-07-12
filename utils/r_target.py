
import numpy as np
from add_target import add_target_pixel
import matplotlib.pyplot as plt
'''
制作三维数据集：同一深度；20*20
'''

def train_data_generate(HSI_curve,test_label,x1,x2,y1,y2,r,h,label):

    for i in range(x1, x2):
        delta = i * 100
        for j in range(y1, y2):
            HSI_curve[j + delta] = add_target_pixel(r, h, j+delta)
            test_label[j + delta] = label

    return HSI_curve,test_label


if __name__ == '__main__':
        # read HSI all curve from npy file
        HSI_curve = np.load(r"D:\ZPEAR\Experiment_data\SUTDF\data\Simulation_data\all_curve.npy") #(10000, 176)
        # print(len(HSI_curve))
        r_b_Iron = np.load(r'D:\ZPEAR\Experiment_data\SUTDF\data\Simulation_data\train_data\r_b_Iron.npy')
        r_b_Nylon_Carpet = np.load(r'D:\ZPEAR\Experiment_data\SUTDF\data\Simulation_data\train_data\r_b_Nylon_Carpet.npy')
        r_b_Plastic_PETE = np.load(r'D:\ZPEAR\Experiment_data\SUTDF\data\Simulation_data\train_data\r_b_Plastic_PETE.npy')
        r_b_Wood_Beam = np.load(r'D:\ZPEAR\Experiment_data\SUTDF\data\Simulation_data\train_data\r_b_Wood_Beam.npy')


        label = test_label = np.zeros(10000)
        train_curve1,train_label1 = train_data_generate(HSI_curve,test_label,20,40,20,40,r_b_Iron,4,1)
        train_curve2,train_label2 = train_data_generate(train_curve1,train_label1,20,40,60,80,r_b_Nylon_Carpet,4,2)
        train_curve3,train_label3 = train_data_generate(train_curve2, train_label2,60,80,20,40, r_b_Plastic_PETE,4,3)
        train_data,train_label = train_data_generate(train_curve3,train_label3,60,80,60,80, r_b_Wood_Beam,4,4)
        train_data = train_data.reshape((100,100,-1))
        train_label = train_label.reshape((100,100))
        print(train_data.shape)
        print(train_label.shape)
        np.save(r'D:\ZPEAR\Experiment_data\HybridSN\train_data_4m.npy',train_data)
        np.save(r'D:\ZPEAR\Experiment_data\HybridSN\train_label_4m.npy',train_label)

        # img = train_label.reshape(100,100)
        # plt.imshow(img)
        # plt.show()



