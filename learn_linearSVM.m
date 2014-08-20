% Using pegasos algorithm to solve a simple linear SVM classifier without 
% a bias term (b)
% Note that we use CVX libarary to first optimal value, and then report the
% convergence rate for pegasos algorithm

clear;
clc;

% load data
load q1x.dat
load q1y.dat

% define variables
X = q1x;
Y = 2*(q1y-0.5);

[train_num, feature_num] = size(X);


lambda = 0.01; % it can be set as 1/C
cvx_begin quiet
    variable w(feature_num);
    variable xi(train_num);
    
    minimize( 0.5*lambda*w'*w + (1 / train_num) *sum(xi));
    subject to
        Y .* (X*w) >= 1 - xi;
        xi >= 0;
cvx_end

fbest = cvx_optval;

maxIter = 5000;
[w, fval] = pegasos(X,Y,lambda,[],maxIter,10);

% reporting accuracy
t_num = sum(sign(X * w) == Y);
accuracy = 100 * t_num / train_num;
fprintf('Accuracy on training set is %.4f %%\n', accuracy);

% plot convergence rate
figure(1), clf
step = 100;
semilogy( 1:step:maxIter, fval(1:step:end) - fbest, 'r-.','LineWidth',1.5 );
xlabel('# iter');
ylabel('f - fmin');
axis([1 maxIter 1e-8 2e2]);
title('Convergence rate for pegasos algorithm');

% visualize
figure(2), clf
xp = linspace(min(X(:,1)), max(X(:,1)), 100);
yp = - w(1) * xp / w(2);
yp1 = - (w(1)*xp - 1) / w(2); % margin boundary for support vectors for y=1
yp0 = - (w(1)*xp + 1) / w(2); % margin boundary for support vectors for y=0

% index of negative samples
idx0 = find(q1y==0);
% index of positive samples
idx1 = find(q1y==1);

plot(q1x(idx0, 1), q1x(idx0, 2), 'rx'); 
hold on
plot(q1x(idx1, 1), q1x(idx1, 2), 'go');
plot(xp, yp, '-b', xp, yp1, '--g', xp, yp0, '--r');
hold off
title(sprintf('decision boundary for a linear SVM classifier with lambda = %g', lambda));
