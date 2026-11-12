
%% EECE5644 Assignment 3 - Question 2
% previous code used: generateDataFromGMM.m.
% EM structure adapted from GMM examples.
% AI-assisted: cross-validation, repetitions, safeguards,
% checkpointing, and visualization.

clear; clc; close all;

Nvalues = [10 100 1000];
orders = 1:10;
K = 10;
R = 100;
starts = 2;
maxIter = 50;
reg = 0.05;
checkpoint = 'Q2_checkpoint.mat';

gmm.priors = [0.30 0.25 0.25 0.20];
gmm.meanVectors = [-3 -2.3 2 3; -2 -1.6 2 -1];
gmm.covMatrices(:,:,1) = [1 .3; .3 .8];
gmm.covMatrices(:,:,2) = [.9 .2; .2 .9];
gmm.covMatrices(:,:,3) = [.8 -.2; -.2 1.2];
gmm.covMatrices(:,:,4) = [1.1 .1; .1 .7];

counts = zeros(3,10);
trainSum = zeros(3,10);
valSum = zeros(3,10);
completed = zeros(3,1);

if isfile(checkpoint)
    S = load(checkpoint);
    if ~isequal(S.Nvalues,Nvalues) || ...
       ~isequal(S.orders,orders) || ...
       S.starts~=starts || S.maxIter~=maxIter || ...
       S.reg~=reg || S.K~=K
        error('Checkpoint settings differ. Rename old checkpoint.');
    end
    counts = S.counts;
    trainSum = S.trainSum;
    valSum = S.valSum;
    completed = S.completed;
    rng(S.randomState);
    fprintf('Resuming from checkpoint: [%d %d %d]\n',completed);
else
    rng(42);
end

[x,labels] = generateDataFromGMM(1000,gmm);
figure;
scatter(x(1,:),x(2,:),12,labels,'filled');
xlabel('x_1'); ylabel('x_2');
title('True Four-Component GMM');
grid on;
exportgraphics(gcf,'Q2_TrueGMM.png');

overallTimer = tic;

for a = 1:length(Nvalues)
    N = Nvalues(a);
    datasetTimer = tic;

    fprintf('\nStarting N=%d (%d/%d completed)\n', ...
        N,completed(a),R);

    for r = completed(a)+1:R
        [x,~] = generateDataFromGMM(N,gmm);
        X = x';

        foldID = zeros(N,1);
        p = randperm(N);
        foldID(p) = mod(0:N-1,K)+1;

        trainLL = zeros(1,10);
        valLL = zeros(1,10);

        for M = orders
            totalTrain = 0;
            totalVal = 0;

            for k = 1:K
                Xt = X(foldID~=k,:);
                Xv = X(foldID==k,:);

                bestLL = -Inf;
                bestModel = [];

                for s = 1:starts
                    model = fitEM(Xt,M,maxIter,reg);
                    ll = sum(log(max(pdf(model,Xt),realmin)));

                    if ll > bestLL
                        bestLL = ll;
                        bestModel = model;
                    end
                end

                totalTrain = totalTrain+bestLL;
                totalVal = totalVal+ ...
                    sum(log(max(pdf(bestModel,Xv),realmin)));
            end

            trainLL(M) = totalTrain/(N*(K-1));
            valLL(M) = totalVal/N;
        end

        [~,bestM] = max(valLL);
        counts(a,bestM) = counts(a,bestM)+1;
        trainSum(a,:) = trainSum(a,:)+trainLL;
        valSum(a,:) = valSum(a,:)+valLL;
        completed(a) = r;

        randomState = rng;
        save(checkpoint,'counts','trainSum','valSum', ...
            'completed','randomState','Nvalues','orders', ...
            'starts','maxIter','reg','K');

        elapsed = toc(datasetTimer);
        runCount = r-max(0,completed(a)-1);
   
        fprintf('N=%d | %d/%d | M=%d | elapsed %.1f min\n', ...
            N,r,R,bestM,elapsed/60);
    end
end

fprintf('\nTotal session time: %.2f hours\n', ...
    toc(overallTimer)/3600);

%% Final results
frequency = counts ./ max(completed,1);
meanTrainLL = trainSum ./ max(completed,1);
meanValLL = valSum ./ max(completed,1);

disp('Selection frequencies:');
disp(array2table(frequency, ...
    'VariableNames',cellstr(compose('M%d',orders)), ...
    'RowNames',cellstr(compose('N%d',Nvalues))));

disp('Mean training log-likelihood:');
disp(meanTrainLL);

disp('Mean validation log-likelihood:');
disp(meanValLL);

figure;
bar(orders,frequency','grouped');
xlabel('Number of GMM Components');
ylabel('Selection Frequency');
legend('N=10','N=100','N=1000');
title('GMM Model Order Selection');
grid on;
exportgraphics(gcf,'Q2_SelectionFrequencies.png');

figure;
tiledlayout(1,2);
nexttile;
plot(orders,meanTrainLL','-o');
title('Training Log-Likelihood');
xlabel('Model Order'); grid on;

nexttile;
plot(orders,meanValLL','-o');
title('Validation Log-Likelihood');
xlabel('Model Order'); grid on;
legend('N=10','N=100','N=1000');

exportgraphics(gcf,'Q2_LogLikelihood.png');

save('Q2_results.mat','frequency','counts', ...
    'meanTrainLL','meanValLL','gmm','completed');

disp('Question 2 completed successfully.');

function model = fitEM(X,M,maxIter,reg)

[N,d] = size(X);
alpha = ones(1,M)/M;
mu = X(randi(N,M,1),:);
baseCov = cov(X,1)+reg*eye(d);
Sigma = repmat(baseCov,1,1,M);
previousLL = -Inf;

for t = 1:maxIter
    joint = zeros(N,M);

    for j = 1:M
        joint(:,j) = alpha(j)* ...
            mvnpdf(X,mu(j,:),Sigma(:,:,j));
    end

    joint = max(joint,realmin);
    total = sum(joint,2);
    w = joint./total;
    ll = sum(log(total));

    if t>1 && abs(ll-previousLL)<1e-5*(1+abs(previousLL))
        break;
    end
    previousLL = ll;

    Nk = sum(w,1);
    alpha = (Nk+1e-6)/(N+M*1e-6);

    for j = 1:M
        if Nk(j)<1e-6
            mu(j,:) = X(randi(N),:);
            Sigma(:,:,j) = baseCov;
        else
            mu(j,:) = (w(:,j)'*X)/Nk(j);
            D = X-mu(j,:);
            Sigma(:,:,j) = ...
                (D'*(D.*w(:,j)))/Nk(j)+reg*eye(d);
        end
    end
end

model = gmdistribution(mu,Sigma,alpha);
end
