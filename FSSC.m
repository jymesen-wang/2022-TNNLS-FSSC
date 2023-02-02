function [result,labelnew,t,Rank,rp] = FSSC(X,B,M,label,alpha_u,alpha_l,isW)
% Input:
% —————— X: data matrix n*d
% —————— B: anchor graph matrix n*m
% —————— M: anchor data matrix m*d
% —————— label: ground truth (for compute clustering results)
% —————— alpha_u: the coefficient of detecing novel class, default 0.99
% —————— alpha_l: the coefficient of changing primal label, default 0
% —————— isW: option '1' means 'compute W for sample similarity', option
% '0' means 'not compute W for reduce space complexity', default 1

% Output:
% —————— result: clustering results ACC NMI ARI
% —————— labelnew: predicted label by FSSC n*1
% —————— t: running time
% —————— Rank: the index of anchors from full samples m*1
% —————— rp: the index of c representative points from full samples c*1 


% alpha_u, alpha_l, alpha_uc and alpha_lc are the regularization parameter for FSSF

% 'Fast Self-Supervised Clustering by Anchor Graph', IEEE Transactions on
% Neural Networks and Learning Systems, 33(9), pp. 4199-4121, 2022. by Zhenyu Ma.
if nargin<7
    isW = 1;
end

if nargin<5
    alpha_u = 0.99;   % =1 no detect novel class, <1 detect novel class
end                   % The larger of alpha_u, the stronger of detecting novel class

if nargin<6
    alpha_l = 0;      % =0 no change of primal label, >0 primal label can change
end
c = length(unique(label));
[n,m] = size(B);
alpha_lc = 0;         % final LP
alpha_uc = 1;         % final LP



FSSC_start = tic;
%% find the sort for each anchor 
Dis = EuDist2(X,M,0);  % O(nmd)
[~,IDx] = sort(Dis);
Rank = IDx(1,:);Rank = Rank';
sRank = sort(Rank);
for i = 1:m-1
    if sRank(i+1) == sRank(i)
        repval = sRank(i+1);
        repidx = find(Rank==repval);
        for j = 2:n
            if ismember(IDx(j,i+1),Rank) == 0
                Rank(repidx(1)) = IDx(j,repidx(1)); % unrepeated idx of all anchors
                break;
            end
        end
    end
end
    
reRank = setdiff(1:1:n,Rank);  % the sort for unlabeled samples

%% Y
Y_m = zeros(n,m+1);
for i = 1:m
    Y_m(Rank(i),i) = 1;        % labels of labeled samples(anchors)
end
for j =1:n
    if ismember(j,Rank) == 0
       Y_m(j,m+1) = 1;         % labeld of unlabeled samples
    else
    end
end
delta1 = 1./sum(B);
delta2 = diag(sum(B));
Bdelta = B.*repmat(delta1,n,1);  % O(mn)
if isW == 1
    W = Bdelta*B';
end

%% alpha and beta
alpha = zeros(n,1);
alpha(Rank) = alpha_l;alpha(reRank) = alpha_u;
beta = ones(n,1)-alpha;
%% F_soft
Balpha = B'.*repmat(alpha',m,1);  % O(nm)
Ybeta = repmat(beta,1,m+1).*Y_m;  % O(n(m+1))
P_c = delta2-Balpha*B;            % O(nm*m)+O(mn)
P_1_c = Balpha'/(P_c);            % O(m^3)+O(nm*m)
F_soft = P_1_c*(B'*Ybeta)+Ybeta;  % O(nm(m+1))+O(nm(m+1))+O(n(m+1))
F_soft = F_soft(:,1:m);


%% special selection
fea = sum(F_soft,2);              % score(i) is the feature for xi
rp = zeros(c,1);                  % Sort_c is the idx of c representative points
if isW == 1
    rate = 1/(max(max(W)));
else
    rate = 1/(max(max(B)));
end
for h = 1:c
    [~,Ic] = sort(fea,'descend');
    rp(h) = Ic(1);
    for l = 1:n
        if isW == 1
            fea(l) = (1 - rate*W(l,Ic(1)))*fea(l);                   % O(nc)
        else
            fea(l) = (1 - rate*sum(Bdelta(l,:).*B(Ic(1),:)))*fea(l); % O(nc)
        end
    end
end


%% Label Propagation Processing
rerp = setdiff(1:1:n,rp);
Y_c = zeros(n,c+1);
for i = 1:c
    Y_c(rp(i),i) = 1;
end
alpha_c = zeros(n,1);
alpha_c(rp) = alpha_lc;alpha_c(rerp) = alpha_uc;
beta_c = ones(n,1)-alpha_c;


%% F_c
Balpha_c = B'.*repmat(alpha_c',m,1);   % O(nm)
Ycbeta = repmat(beta_c,1,c+1).*Y_c;    % O(n(c+1))
P_c = delta2 - Balpha_c*B;             % O(nm*m)+O(mn)
P_1_c = Balpha_c'/(P_c);               % O(m^3)+O(nm*m)
F_c = P_1_c*(B'*Ycbeta)+Ycbeta;        % O(nm(c+1))+O(nm(c+1))+O(n(c+1))
F_c = F_c(:,1:c);

%% Label and Results
[~,labelnew] = max(F_c,[],2);
t = toc(FSSC_start);
result(1:7) = ClusteringMeasure_All(label,labelnew);
result = [result(1:2) result(7)];