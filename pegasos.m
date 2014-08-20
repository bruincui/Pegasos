function [w, fval] = pegasos( X,Y,lambda,w0,maxIter,k )
%
% Input:
% X: n*d matrix, n is the number of training samples, d is the dimension of
% features;
% Y: n*1 vector indicating class labels (1 or -1) for samples in X;
% w0: the initial value of w, it's a column vector
% lambda, k: parameters in Pegasos algorithm (default: lambda=1, k=0.1*n);
% maxIter: maximum number of iterations for w convergence; (default:
% 10000); Training stops if maxIter is satisfied;
% 
% Output:
% w: weight vector in SVM primal problem:
% fval: objective value in each iteration
%
% References:
% [1] Pegasos-Primal Estimated sub-Gradient SOlver for SVM


[n,d] = size(X);
if(size(Y,1) ~= n)
    error('Number of samples in X and Y must be same!');
end

if(sum(Y~=1 & Y~=-1) > 0)
    error('The elements in Y must be 1 or -1!');
end

if(nargin<3 || isempty(lambda)),  lambda = 1;  end
if(nargin<4), w0 = []; end
if(nargin<5 || isempty(maxIter)),   maxIter = 10000;  end
if(nargin<6 || isempty(k)), k = ceil(0.1*n);    end

if k > n
    k = n;
end

% intialization
if isempty(w0)
    w = zeros(d,1);
else
    if any(size(w0) ~= [d,1])
        error('The dimension of initial value is not correct!');
    else
        w = w0;
    end
end

% objective value in each iteration
fval = zeros(maxIter, 1);

for t = 1:maxIter
    %if mod(t,2000) == 0
    %    fprintf('#%d Iter~\n',t);
    %end
    
    % generating indexes uniformly at random without repetitions
    idx = randperm(n);
    idx = idx(1:k);
    A_t = X(idx,:);
    y_t = Y(idx,:);
    
    % indexes with violated samples
    idx = (y_t .* (A_t * w)) < 1;
    eta_t = 1 / (lambda * t);
    
    % update w
    f_dis = A_t(idx,:) .* repmat(y_t(idx,:),1,d);
    w = (1 - 1/t) * w + (eta_t / k) * sum(f_dis, 1)';
    
    % optional operation: project w into a ball
    w = min(1, 1 / (sqrt(lambda) * norm(w))) * w;
    
    % recording objective value with current w
    margin = 1 - (Y .* (X * w));
    idx = margin > 0;
    fval(t) = 0.5 * lambda * w' * w + (1/n) * sum(margin(idx));
end

end

