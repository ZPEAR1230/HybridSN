# HybridSN
混合2-D和3-D卷积实现高光谱分类
先验知识：1）2-D-CNN无法处理数据的第三维度——光谱信息
         2） 只使用3D-卷积，虽然可以提取第三维——光谱维度的特征，能同时进行空间和空间特征表示，但数据计算量特别大，且对特征的表现能力比较差（因为许多光谱带上的纹理是相似的）
结论：将空间光谱和光谱的互补信息分别以3D-CNN和2D-CNN层组合到了一起，从而充分利用了光谱和空间特征图，来克服以上缺点。

参考论文：《HybridSN: Exploring 3-D–2-DCNN Feature Hierarchy for Hyperspectral Image Classification》
