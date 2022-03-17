function [X1,X2,U] = BKHK_onestep(X0)
% U is the Anchor set:2*d
% X is the data matrix:n*d
%% initialization by k-means
[n,d] = size(X0);
[~,C] = kmeans(X0,2);
C1 = C(1,:);C2 = C(2,:);
for i = 1:d
    if C1(i) < C2(i)
        C(1,:) = C2;
        C(2,:) = C1;
        break
    elseif C1(i) > C2(i)
        break
    else
    end
end
% [C,~] = fcm(X0,2);
%% compute distance to build E:n*2
E = L2_distance_1(X0',C');
%% set the amount of samples in two clusters:u and calculate e,g and G
if rem(n,2) == 0
    u = n/2;
else
    u = (n-1)/2;
end                            % get the u and n-u
g = zeros(n,1);
e1 = E(:,1);
e2 = E(:,2);
e = e1-e2;                     % calculate the vector e:n*1     
[~,r] = sort(e);               % sort the e to get index
v = r(1:u);
w = r(u+1:n);
g(v) = 1;                      % calculate the vector g:n*1
G = [g,ones(n,1)-g];           % set G:n*2
%% calculate the cluster center for both of the two clusters
X1 = X0(v,:);
X2 = X0(w,:);                  % X is the samples according to the index
S1 = zeros(u,1);
S2 = zeros(n-u,1);
% for i = 1:u
%     for j = 1:u
%         b = L2_distance(X1(i,:),X1(j,:));
%         S1(i) = S1(i)+b;
%     end
% end
w1 = mean(X1);                 % get mean of each sample
for i = 1:u
    S1(i) = norm(X1(i,:)-w1);
end
[~,r1] = min(S1);              % select minmum of distance to get center of cluster1
% I1 = v(r1);                  % problem about rank
% for i = 1:n-u
%     for j = 1:n-u
%         b = L2_distance(X2(i,:),X2(j,:));
%         S2(i) = S2(i)+b;
%     end
% end
w2 = mean(X2);
for i = 1:n-u
    S2(i) = norm(X2(i,:)-w2);
end
[~,r2] = min(S2);              % select minmum of distance to get center of cluster1
% I2 = w(r2);
%% get the Anchor set of onestep of BKHK
U = [X1(r1,:);X2(r2,:)];