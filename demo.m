DataName = 'USPS';
load([DataName,'_data.mat'])
numAnchor = 9;     % m = 2^(numAnchor) Letter 10 USPS 9
numNeighbor = 20;   % Letter 20 USPS 20
generateAnchor = 1; % BKHK

% Construct Anchor Graph B (n*m) and Anchor data M (m*d)
% generateAnchor = 1 : BKHK 2 : kmeans++ 3: kmeans 4: Random Selection  1 as default
[B,M] = AnchorGEN(X,numAnchor,numNeighbor,generateAnchor);  

% Operate FSSC
[result,labelnew,t,Rank,rp] = FSSC(X,B,M,label);

% result - clustering result [ACC NMI ARI];
%   t    - running time