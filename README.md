# FSSC

Using the code, please cite:
Wang J, Ma Z, Nie F, et al. Fast Self-Supervised Clustering with Anchor Graph[J]. IEEE Transactions on Neural Networks and Learning Systems, 33(9), pp. 4199-4212, 2022.

https://ieeexplore.ieee.org/document/9354504

The code explanation: 
The main function of the code: ULGEmzy.m and FSSC.m
You can use demo.m to perform FSSC clustering on USPS and Letter data sets. 

To use function 'ULGEmzy.m', please follow the input/output format:

[B,M] = ULGEmzy(X,numAnchor,numNeighbor,generateAnchor); 

# Input:
———— numAnchor: the number of anchors
———— numNeighbor: the number of neighbors of anchor graph
———— generateAnchor: option '1' means 'BKHK', '2' means 'kmeans', '3' means 'litekmeans'


If you have any questions, please connect zhenyu.ma@mail.nwpu.edu.cn
