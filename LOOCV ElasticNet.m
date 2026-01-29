%% =========================================================
%  HON-Pearson FC + LOOCV Elastic Net
%% =========================================================
clear; clc; close all;

%% Paths
projectRoot = fileparts(fileparts(mfilename('fullpath')));
dataDir     = fullfile(projectRoot,'data','Alzheimer_fnets');
resultsDir  = fullfile(projectRoot,'results');
if ~exist(resultsDir,'dir'), mkdir(resultsDir); endx

%% Groups
Groups = { ...
    struct('name','CN','path','45_CN_yeni','label',0), ...
    struct('name','AD','path','45_AD_yeni','label',1) ...
};

%% ROI & features
ROI_Ind = [1:200 211:410];
ROI_N   = numel(ROI_Ind);
UT_idx  = triu(true(ROI_N),1);
FisherZ = @(x) atanh(max(min(x,0.999999),-0.999999));

%% Feature extraction
X_ALL = [];
y_ALL = [];

for g = 1:numel(Groups)

    files = dir(fullfile(dataDir,Groups{g}.path,'*.txt'));

    for i = 1:numel(files)
        try
            M = readmatrix(fullfile(files(i).folder,files(i).name));
            M = M(ROI_Ind,ROI_Ind);
            M(~isfinite(M)) = 0;
            M(1:ROI_N+1:end) = 0;

            Mz  = FisherZ(M);
            HON = corr(Mz,'Type','Pearson','Rows','pairwise');
            HON(~isfinite(HON)) = 0;
            HON(1:ROI_N+1:end) = 0;

            X_ALL = [X_ALL; FisherZ(HON(UT_idx))'];
            y_ALL = [y_ALL; Groups{g}.label];

        catch
            continue;
        end
    end
end

X_ALL(~isfinite(X_ALL)) = 0;

save(fullfile(resultsDir,'HON_Pearson_Upper.mat'), ...
     'X_ALL','y_ALL','ROI_Ind','UT_idx','-v7.3');

%% LOOCV + Elastic Net
[N,~] = size(X_ALL);

alpha   = 0.5;
lambdaN = 30;

pred  = zeros(N,1);
score = zeros(N,1);

for i = 1:N

    tr = setdiff(1:N,i);
    te = i;

    [B,Fit] = lassoglm( ...
        X_ALL(tr,:), y_ALL(tr),'binomial', ...
        'Alpha',alpha, ...
        'NumLambda',lambdaN, ...
        'CV',3, ...
        'Standardize',true);

    b  = Fit.IndexMinDeviance;
    p  = 1 ./ (1 + exp(-(X_ALL(te,:)*B(:,b) + Fit.Intercept(b))));

    score(i) = p;
    pred(i)  = p >= 0.5;
end

%% Performance
ACC = mean(pred == y_ALL);
[~,~,~,AUC] = perfcurve(y_ALL,score,1);

C = confusionmat(y_ALL,pred);
TP=C(2,2); TN=C(1,1); FP=C(1,2); FN=C(2,1);

Sensitivity = TP/(TP+FN);
Specificity = TN/(TN+FP);
Precision   = TP/(TP+FP);
F1          = 2*(Precision*Sensitivity)/(Precision+Sensitivity);

fprintf('\nACC: %.2f%% | AUC: %.3f | SEN: %.2f%% | SPE: %.2f%% | F1: %.3f\n', ...
        ACC*100, AUC, Sensitivity*100, Specificity*100, F1);

figure('Color','w');
confusionchart(C,{'CN','AD'}, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
