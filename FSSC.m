function [U_final,Rank,labelnew,rp,result] = FSSC(X,q,k0,label,rate,alpha_u,alpha_l)



% X : data matix n by d ; q is hierarchy
% m = 2^(q-1) is the number of anchors; c is class
% k0 is nearest neighbor for bipartite graph
% sigma is the regularization parameter for semi-supervised framework
% alpha_l = 0; % 0时原标签不可变，大于0时原标签可变
% alpha_u = 0.99; % 等于1时不发现新类，小于1时发现新类，且值越小发现新类的能力就越强
% U_final为锚点集
% Rank为锚点序号
% labelnew为FSSC聚类结果
% rp为代表点序号
% result为聚类结果评估指标包含ACC,NMI,Purity,BKHK运行时间以及整体运算时间，1*5
if nargin < 7
    alpha_l = 0;
end
if nargin < 6
    alpha_u = 0.99;
end
if nargin < 5
    rate = 10;
end


alpha_lc = 0; % 0时原标签不可变，大于0时原标签可变
alpha_uc = 1; % 等于1时不发现新类，小于1时发现新类，且值越小发现新类的能力就越强
% q = 10;
% k0 = 30;
c = length(unique(label));
dd_first = 1;
dd_second = 1;
result = zeros(1,5);
tic
pause(1)
%% Main BKHK
m = 2^(q-1);    % 锚点数量
A = cell(m,q);  % 构造元胞数组储存每次2分类的结果
A{1,1} = X; 
[n,d] = size(X);
for i = 1:(q-1)
    for j = 1:2^(i-1)
        [A{2*j-1,i+1},A{2*j,i+1},~] = BKHK_onestep(A{j,i});   % 时间复杂度O(ndlog(m))
        % 第一次二分的时间复杂度为2nd（d为数据的维度，n为样本个数），下一层的单次二分kmeans的样本数都会减半，但二分kmeans的次数会加倍
        % 因此每一层的二分Kmeans的时间复杂度相同，假设m为最终选取的锚点个数，因此执行二分Kmeans的层数为log(m),由于每一层的时间复杂度
        % 均为2nd,因此最终的时间复杂度为O(ndlog(m)) (常数2被省略掉了)
    end
end
U_final = zeros(m,d);     % build the U_final matrix
% get the rank of anchors in the full samples and calculate U_final 
Rank = zeros(m,1);
% num_sample = 0; 
for i = 1:2^(q-2)   % 时间复杂度为2nd（相当于某一层的二分Kmeans复杂度），因此时间复杂度为O(nd)(常数2被省略掉了)
    [A{2*i-1,q},A{2*i,q},U] = BKHK_onestep(A{i,q-1}); 
    U_final(2*i-1,:) = U(1,:);
    U_final(2*i,:) = U(2,:);
end
% find the sort for each anchor 
for i = 1:m        % 时间复杂度为O(nmd)
    for j = 1:n
        if U_final(i,:) == X(j,:) % d次元素判断
            Rank(i) = j;
        else
        end
    end
end
% calculate the complement of Rank(the sort for unlabeled samples)
H = zeros(n,1);
for i = 1:n
    H(i) = i;
end
reRank = setdiff(H,Rank);
% tic
t1 = toc;
%% Construct Bipartite graph
n = size(X,1);
E_distance_m = L2_distance_1(X',U_final');   % 构建锚点和样本点的距离矩阵的时间复杂度为O(nmd)
[~,idx] = sort(E_distance_m,2);% sort each row
gamma = zeros(n,1);       % different samples have different gamma
for i = 1:n               % 某一个样本如果是选取的锚点，该行的第一项肯定是0，取2：k+2
     if ismember(i,Rank)==0
        id = idx(i,1:k0+1);
        di = E_distance_m(i,id);
        gamma(i) = (k0*di(k0+1)-sum(di(1:k0))+eps)/2;
     else
        id = idx(i,2:k0+2);
        di = E_distance_m(i,id);
        gamma(i) = (k0*di(k0+1)-sum(di(1:k0))+eps)/2;
     end
end
B = zeros(n,m);
for i = 1:n
    [B(i,:),~] = EProjSimplex_new(-E_distance_m(i,:)/(2*gamma(i)));
end

%% Solve SSL to get soft label matrix F_soft 
%% Y
Y_m = zeros(n,m+1);
for i = 1:m
    Y_m(Rank(i),i) = 1;      % labels of labeled samples(anchors)
end
for j =1:n
    if ismember(j,Rank) == 0
       Y_m(j,m+1) = 1;      % labeld of unlabeled samples
    else
    end
end
% Y_initial_m = Y_m;
%delta
delta = diag(sum(B));
W = (B/(delta))*B';
% L = diag(sum(W)) - W;
%% Objective Function
% Object_Func = zeros(max(dd_first,dd_second),2);
%% alpha and beta
alpha = zeros(n,1);alpha(Rank) = alpha_l;alpha(reRank) = alpha_u;
alpha = diag(alpha);beta = eye(n)-alpha;
%% F
P_c = delta - B'*alpha*B;    %% mn mn m^2*n时间复杂度是m^2*n
P_1_c = (alpha*B)/(P_c);     %% mn m^3
for i = 1:dd_first
    F_soft = P_1_c*(B'*(beta*Y_m))+beta*Y_m;  % 时间复杂度为m*m*n 具体为(m*m+m)*n
    Y_m = F_soft;
end
F_soft = F_soft(:,1:m);
%% special selection
fea = sum(F_soft,2);          % score(i) is the feature for xi
rp = zeros(c,1);              % Sort_c is the idx of c representative points
rate_lim = 1/(max(max(W)));   % 相似度修正倍率的最大值
if rate > rate_lim
    rate = rate_lim;          % 给定rate若超过最大值，令rate为最大值
end
for h = 1:c
    [~,Ic] = sort(fea,'descend');
    rp(h) = Ic(1);
    for l = 1:n
         fea(l) = (1 - rate*W(l,Ic(1)))*fea(l);      % modify and update all features 由于生成的W矩阵各个元素值都太小，在更新
        % 时可以考虑利用倍数来加大feature之间的差距
    end
end
%% Label Propagation Processing
rerp = setdiff(H,rp);
Y_c = zeros(n,c+1);
for i = 1:c
    Y_c(rp(i),i) = 1;
end
% Y_initial = Y_c;
alpha_c = zeros(n,1);alpha_c(rp) = alpha_lc;alpha_c(rerp) = alpha_uc;
alpha_c = diag(alpha_c);
beta_c = eye(n)-alpha_c;
%% F_c
P_c = delta - B'*alpha_c*B;  %% mn mn m^2*n时间复杂度是m^2*n
P_1_c = (alpha_c*B)/(P_c);   %% mn m^3
for i = 1:dd_second
    F_c = P_1_c*(B'*(beta_c*Y_c))+beta_c*Y_c; % 时间复杂度为m*m*n 具体为(m*m+m)*n
    Y_c = F_c;
end
F_c = F_c(:,1:c);
[~,index] = sort(transpose(F_c),'descend');
labelnew = index(1,:);
labelnew = labelnew';
t2 = toc;
t3 = t2 - t1;
result(1:3) = ClusteringMeasure(label,labelnew);
result(4) = t3;
result(5) = t2;