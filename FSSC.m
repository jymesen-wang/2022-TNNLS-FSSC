function [U_final,Rank,labelnew,rp,result] = FSSC(X,q,k0,label,rate,alpha_u,alpha_l)



% X : data matix n by d ; q is hierarchy
% m = 2^(q-1) is the number of anchors; c is class
% k0 is nearest neighbor for bipartite graph
% sigma is the regularization parameter for semi-supervised framework
% alpha_l = 0; % 0ʱԭ��ǩ���ɱ䣬����0ʱԭ��ǩ�ɱ�
% alpha_u = 0.99; % ����1ʱ���������࣬С��1ʱ�������࣬��ֵԽС���������������Խǿ
% U_finalΪê�㼯
% RankΪê�����
% labelnewΪFSSC������
% rpΪ��������
% resultΪ����������ָ�����ACC,NMI,Purity,BKHK����ʱ���Լ���������ʱ�䣬1*5
if nargin < 7
    alpha_l = 0;
end
if nargin < 6
    alpha_u = 0.99;
end
if nargin < 5
    rate = 10;
end


alpha_lc = 0; % 0ʱԭ��ǩ���ɱ䣬����0ʱԭ��ǩ�ɱ�
alpha_uc = 1; % ����1ʱ���������࣬С��1ʱ�������࣬��ֵԽС���������������Խǿ
% q = 10;
% k0 = 30;
c = length(unique(label));
dd_first = 1;
dd_second = 1;
result = zeros(1,5);
tic
pause(1)
%% Main BKHK
m = 2^(q-1);    % ê������
A = cell(m,q);  % ����Ԫ�����鴢��ÿ��2����Ľ��
A{1,1} = X; 
[n,d] = size(X);
for i = 1:(q-1)
    for j = 1:2^(i-1)
        [A{2*j-1,i+1},A{2*j,i+1},~] = BKHK_onestep(A{j,i});   % ʱ�临�Ӷ�O(ndlog(m))
        % ��һ�ζ��ֵ�ʱ�临�Ӷ�Ϊ2nd��dΪ���ݵ�ά�ȣ�nΪ��������������һ��ĵ��ζ���kmeans��������������룬������kmeans�Ĵ�����ӱ�
        % ���ÿһ��Ķ���Kmeans��ʱ�临�Ӷ���ͬ������mΪ����ѡȡ��ê����������ִ�ж���Kmeans�Ĳ���Ϊlog(m),����ÿһ���ʱ�临�Ӷ�
        % ��Ϊ2nd,������յ�ʱ�临�Ӷ�ΪO(ndlog(m)) (����2��ʡ�Ե���)
    end
end
U_final = zeros(m,d);     % build the U_final matrix
% get the rank of anchors in the full samples and calculate U_final 
Rank = zeros(m,1);
% num_sample = 0; 
for i = 1:2^(q-2)   % ʱ�临�Ӷ�Ϊ2nd���൱��ĳһ��Ķ���Kmeans���Ӷȣ������ʱ�临�Ӷ�ΪO(nd)(����2��ʡ�Ե���)
    [A{2*i-1,q},A{2*i,q},U] = BKHK_onestep(A{i,q-1}); 
    U_final(2*i-1,:) = U(1,:);
    U_final(2*i,:) = U(2,:);
end
% find the sort for each anchor 
for i = 1:m        % ʱ�临�Ӷ�ΪO(nmd)
    for j = 1:n
        if U_final(i,:) == X(j,:) % d��Ԫ���ж�
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
E_distance_m = L2_distance_1(X',U_final');   % ����ê���������ľ�������ʱ�临�Ӷ�ΪO(nmd)
[~,idx] = sort(E_distance_m,2);% sort each row
gamma = zeros(n,1);       % different samples have different gamma
for i = 1:n               % ĳһ�����������ѡȡ��ê�㣬���еĵ�һ��϶���0��ȡ2��k+2
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
P_c = delta - B'*alpha*B;    %% mn mn m^2*nʱ�临�Ӷ���m^2*n
P_1_c = (alpha*B)/(P_c);     %% mn m^3
for i = 1:dd_first
    F_soft = P_1_c*(B'*(beta*Y_m))+beta*Y_m;  % ʱ�临�Ӷ�Ϊm*m*n ����Ϊ(m*m+m)*n
    Y_m = F_soft;
end
F_soft = F_soft(:,1:m);
%% special selection
fea = sum(F_soft,2);          % score(i) is the feature for xi
rp = zeros(c,1);              % Sort_c is the idx of c representative points
rate_lim = 1/(max(max(W)));   % ���ƶ��������ʵ����ֵ
if rate > rate_lim
    rate = rate_lim;          % ����rate���������ֵ����rateΪ���ֵ
end
for h = 1:c
    [~,Ic] = sort(fea,'descend');
    rp(h) = Ic(1);
    for l = 1:n
         fea(l) = (1 - rate*W(l,Ic(1)))*fea(l);      % modify and update all features �������ɵ�W�������Ԫ��ֵ��̫С���ڸ���
        % ʱ���Կ������ñ������Ӵ�feature֮��Ĳ��
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
P_c = delta - B'*alpha_c*B;  %% mn mn m^2*nʱ�临�Ӷ���m^2*n
P_1_c = (alpha_c*B)/(P_c);   %% mn m^3
for i = 1:dd_second
    F_c = P_1_c*(B'*(beta_c*Y_c))+beta_c*Y_c; % ʱ�临�Ӷ�Ϊm*m*n ����Ϊ(m*m+m)*n
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