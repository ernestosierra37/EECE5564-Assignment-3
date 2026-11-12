
% EECE5644 - Assignment 3, Question 1

% Gaussian classification adapted from ERMwithClabels.m
% Data generation: generateDataFromGMM.m
% Gaussian evaluation: evalGaussianPDF.m
%
% AI-assisted:
% MLP implementation, 10-fold cross-validation,
% final training, checkpointing, and visualization.
%
clear; clc; close all;


cvFile = 'Q1_CV_results.mat';

if ~isfile(cvFile)
    error('Q1_CV_results.mat not found.');
end

S = load(cvFile);

trainingSizes = S.trainingSizes;
hiddenSizes = S.hiddenSizes;
bestHiddenSizes = S.bestHiddenSizes;
bestCVErrors = S.bestCVErrors;
allCVErrors = S.allCVErrors;

trainData = S.trainData;
trainLabels = S.trainLabels;
x = S.x;
testLabels = S.testLabels;
gmmParameters = S.gmmParameters;

C = 4;
n = 3;
N = size(x,2);

numDatasets = length(trainingSizes);

if any(isnan(allCVErrors(:))) || ...
   any(isnan(bestCVErrors(:))) || ...
   any(bestHiddenSizes(:) == 0)

    error('Cross-validation results are incomplete.');
end

fprintf('Completed cross-validation results loaded.\n');


cvResultsTable = table( ...
    trainingSizes(:), ...
    bestHiddenSizes(:), ...
    100*bestCVErrors(:), ...
    'VariableNames', ...
    {'TrainingSamples','BestP','CVErrorPercent'});

disp(cvResultsTable);


pxgivenl = zeros(C,N);

for l = 1:C

    pxgivenl(l,:) = evalGaussianPDF( ...
        x, ...
        gmmParameters.meanVectors(:,l), ...
        gmmParameters.covMatrices(:,:,l));

end

classScores = pxgivenl .* gmmParameters.priors';

[~,mapDecisions] = max(classScores,[],1);

Perror = mean(mapDecisions(:) ~= testLabels(:));

fprintf('\nOptimal MAP classification error: %.2f%%\n', ...
    100*Perror);

numFinalRestarts = 3;

options = trainingOptions("adam", ...
    MaxEpochs=200, ...
    MiniBatchSize=32, ...
    InitialLearnRate=0.001, ...
    Shuffle="every-epoch", ...
    Verbose=false, ...
    Plots="none");


checkpointFile = 'Q1_final_progress.mat';

if isfile(checkpointFile)

    progress = load(checkpointFile);

    assert(isequal(progress.trainingSizes,trainingSizes) && ...
        isequal(progress.bestHiddenSizes,bestHiddenSizes) && ...
        progress.numFinalRestarts == numFinalRestarts, ...
        'Final-training checkpoint settings do not match.');

    finalNetworks = progress.finalNetworks;
    finalTestErrors = progress.finalTestErrors;
    finalTrainingLosses = progress.finalTrainingLosses;

    fprintf('Final-training checkpoint loaded.\n');

else

    finalNetworks = cell(numDatasets,1);
    finalTestErrors = nan(numDatasets,1);
    finalTrainingLosses = nan(numDatasets,1);

    save(checkpointFile, ...
        'finalNetworks','finalTestErrors', ...
        'finalTrainingLosses','trainingSizes', ...
        'bestHiddenSizes','numFinalRestarts');

    fprintf('Final-training checkpoint initialized.\n');

end



for i = 1:numDatasets

    if ~isnan(finalTestErrors(i)) && ...
       ~isempty(finalNetworks{i})

        fprintf('\nN = %d: Already completed (%.2f%%)\n', ...
            trainingSizes(i),100*finalTestErrors(i));

        continue;

    end

    Ntrain = trainingSizes(i);
    P = bestHiddenSizes(i);

    Xtrain = trainData{i}';
    Ytrain = trainLabels{i}(:);
    Ttrain = zeros(Ntrain,C);

    for k = 1:C
        Ttrain(:,k) = (Ytrain == k);
    end

    bestTrainingLoss = inf;
    bestNet = [];

    fprintf('\nFinal training: N = %d, P = %d\n', ...
        Ntrain,P);

    for restart = 1:numFinalRestarts

        rng(200000*i + restart);

        layersFinal = [
            featureInputLayer(n,Normalization="none")
            fullyConnectedLayer(P)
            eluLayer
            fullyConnectedLayer(C)
            softmaxLayer
        ];

        netFinal = trainnet( ...
            Xtrain,Ttrain, ...
            layersFinal,"crossentropy",options);

        trainScores = predict(netFinal,Xtrain);

        trainScores = max(double(trainScores),1e-12);

        trainingLoss = -mean(sum( ...
            Ttrain .* log(trainScores),2));

        fprintf('  Restart %d: training loss = %.6f\n', ...
            restart,trainingLoss);

        if trainingLoss < bestTrainingLoss

            bestTrainingLoss = trainingLoss;
            bestNet = netFinal;

        end

    end

    finalNetworks{i} = bestNet;
    finalTrainingLosses(i) = bestTrainingLoss;

    Xtest = x';

    testScores = predict(bestNet,Xtest);

    [~,testPredictions] = max(testScores,[],2);

    finalTestErrors(i) = mean( ...
        testPredictions(:) ~= testLabels(:));

    fprintf('  Final test error = %.2f%%\n', ...
        100*finalTestErrors(i));


    save(checkpointFile, ...
        'finalNetworks','finalTestErrors', ...
        'finalTrainingLosses','trainingSizes', ...
        'bestHiddenSizes','numFinalRestarts');

    fprintf('  Final model saved.\n');

end


finalResultsTable = table( ...
    trainingSizes(:), ...
    bestHiddenSizes(:), ...
    100*bestCVErrors(:), ...
    100*finalTestErrors(:), ...
    'VariableNames', ...
    {'TrainingSamples','BestP', ...
     'CVErrorPercent','TestErrorPercent'});

fprintf('\nFinal Question 1 results:\n');

disp(finalResultsTable);

fprintf('Optimal MAP test error: %.2f%%\n', ...
    100*Perror);


figure;

semilogx(trainingSizes, ...
    100*finalTestErrors, ...
    'o-', ...
    'LineWidth',2, ...
    'MarkerSize',8);

hold on;

yline(100*Perror,'--', ...
    'Optimal MAP', ...
    'LineWidth',2);

xlabel('Number of Training Samples');
ylabel('Test Classification Error (%)');

title('MLP Test Error vs Optimal MAP Error');

legend('MLP Test Error', ...
       'Optimal MAP Error', ...
       'Location','best');

grid on;

exportgraphics(gcf,'Q1_Final_TestErrors.png');


figure;

semilogx(trainingSizes, ...
    100*allCVErrors, ...
    'o-', ...
    'LineWidth',1.5);

xlabel('Number of Training Samples');
ylabel('Cross-Validation Classification Error (%)');

title('MLP Cross-Validation Error by Hidden-Layer Size');

legend( ...
    'P = 2', ...
    'P = 5', ...
    'P = 10', ...
    'P = 20', ...
    'Location','best');

grid on;

exportgraphics(gcf,'Q1_CrossValidation_Errors.png');


save('Q1_complete_results.mat', ...
    'finalResultsTable', ...
    'cvResultsTable', ...
    'finalNetworks', ...
    'finalTestErrors', ...
    'finalTrainingLosses', ...
    'trainingSizes', ...
    'hiddenSizes', ...
    'bestHiddenSizes', ...
    'bestCVErrors', ...
    'allCVErrors', ...
    'Perror', ...
    'gmmParameters', ...
    'numFinalRestarts');

fprintf('\nQuestion 1 completed successfully.\n');
