% X:num*dim
%alpha:regulization num
%numanchor:the real number of anchors is 2^numAnchor
%[1] F. Nie, W. Zhu, and X. Li, "Unsupervised Large Graph Embedding," AAAI2017.
%[2] W. Zhu, F. Nie, and X. Li, "Fast Spectral Clustering with Efficient Large
%Graph Construction," ICASSP2017.
function [Z,M] = ULGEmzy(X,numAnchor,numNearestAnchor,selAnchor)

[num,d] = size(X);

if selAnchor == 1 % BKHK
    [~,locAnchor] = hKM(X',[1 :num],numAnchor,1); % here we use BKHK algorithm proposed in [2] to generate anchors.
elseif selAnchor == 2 % kmeans
    [~,locAnchor] = kmeans(X,2^(numAnchor));
    locAnchor = locAnchor';
elseif selAnchor == 3 % litekmeans
    [~,locAnchor] = litekmeans(X,2^(numAnchor));
    locAnchor = locAnchor';
end
M=locAnchor';

Z = ConstructA_NP(X',(locAnchor),numNearestAnchor);
end