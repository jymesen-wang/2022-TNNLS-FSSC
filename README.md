# FSSC

Using the code, please cite:
Wang J, Ma Z, Nie F, et al. Fast Self-Supervised Clustering with Anchor Graph[J]. IEEE Transactions on Neural Networks and Learning Systems, 33(9), pp. 4199-4212, 2022.

https://ieeexplore.ieee.org/document/9354504

The code explanation: 
The main function of the code: ULGEmzy.m and FSSC.m
You can use demo.m to perform FSSC clustering on USPS and Letter data sets. 
If you have any questions, please connect zhenyu.ma@mail.nwpu.edu.cn

To use function 'ULGEmzy.m' for construct anchor graph, please follow the input/output format:

[B,M] = ULGEmzy(X,numAnchor,numNeighbor,generateAnchor); 

Input:
———— numAnchor: the number of anchors
———— numNeighbor: the number of neighbors of anchor graph
———— generateAnchor: option '1' means 'BKHK', '2' means 'kmeans', '3' means 'litekmeans'

Output:
———— B: anchor graph
———— M: anchor data matrix

To use function 'FSSC.m' for self-supervised clustering, please follow the input/output format:

[result,labelnew,t,Rank,rp] = FSSC(X,B,M,label,alpha_u,alpha_l,isW)

Input:
———— X: data matrix
———— B: anchor graph matrix
———— M: anchor data matrix
———— label: ground truth (for compute clustering results)
———— alpha_u: the coefficient of detecing novel class, default 0.99
———— alpha_l: the coefficient of changing primal label, default 0
———— isW: option '1' means 'compute W for sample similarity', option '0' means 'not compute W for reduce space complexity', default 1

Output:
———— result: clustering results ACC NMI ARI
———— labelnew: predicted label by FSSC
———— t: running time
———— Rank: the index of anchors from full samples
———— rp: the index of c representative points from full samples
