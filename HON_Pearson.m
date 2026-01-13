%% =========================================================
%  ALZHEIMER HON-PEARSON DATA PREPARATION (FOR SVM)
%% =========================================================

clear; clc; close all;

%% =========================================================
% 1. SETTINGS
%% =========================================================

fprintf(' Preparing HON-Pearson data for SVM...\n');

ProjectRoot = fileparts(mfilename('fullpath'));
MainFolder  = ProjectRoot;
OutputFolder = fullfile(ProjectRoot, 'ALZHEIMER_SVM_HON_Pearson');
if ~exist(OutputFolder,'dir'), mkdir(OutputFolder); end

% ROI selection (410 → 400)
ROI_Indices = [1:200, 211:410];
ROI_Count = numel(ROI_Indices);

% Parameters
Threshold_Ratio = 0.26;
SafeAtanh = @(x) atanh(max(min(x,0.999999), -0.999999));

%% =========================================================
% 2. DATASET
%% =========================================================

Groups = { ...
    struct('name','CN','path','45_CN_yeni','label',0), ...
    struct('name','AD','path','45_AD_yeni','label',1) ...
};

X_SVM = [];
y_SVM = [];

%% =========================================================
% 3. MAIN PROCESSING LOOP
%% =========================================================

for g = 1:numel(Groups)

    GroupPath = fullfile(MainFolder, Groups{g}.path);
    Files = dir(fullfile(GroupPath, '*.txt'));

    fprintf('  %s group (%d files)\n', Groups{g}.name, numel(Files));

    for i = 1:numel(Files)
        try
            %% --- A) DATA READING ---
            RawMatrix = readmatrix(fullfile(GroupPath, Files(i).name));
            Matrix = RawMatrix(ROI_Indices, ROI_Indices);

            if size(Matrix,1) ~= ROI_Count, continue; end
            Matrix(~isfinite(Matrix)) = 0;
            Matrix(1:ROI_Count+1:end) = 0;

            %% --- B) FISHER-Z TRANSFORMATION (ONCE) ---
            Matrix_Z = SafeAtanh(Matrix);

            %% --- C) HON (PEARSON) ---
            HON = corr(Matrix_Z, 'Type','Pearson','Rows','pairwise');
            HON(~isfinite(HON)) = 0;
            HON(1:ROI_Count+1:end) = 0;

            %% --- D) THRESHOLDING ---
            upper = abs(HON(triu(true(ROI_Count),1)));
            if ~isempty(upper)
                threshold = prctile(upper,(1-Threshold_Ratio)*100);
                HON(abs(HON) < threshold) = 0;
            end

            %% --- E) SVM FEATURES ---
            % NODE STRENGTH COMPUTED FROM HON
            NodeStrength = mean(abs(HON),2)';
            X_SVM = [X_SVM; NodeStrength];
            y_SVM = [y_SVM; Groups{g}.label];

        catch
            continue;
        end
    end
end

%% =========================================================
% 4. DATA CLEANING
%% =========================================================

X_SVM(isnan(X_SVM)) = 0;
ROI_Labels = compose("ROI_%03d",(1:ROI_Count)');
X_MLP = X_SVM;   
y_All = y_SVM;   

%% =========================================================
% 5. SVM DATA SAVING
%% =========================================================

SVM_Data.X   = X_MLP;        % (N x 400)
SVM_Data.y   = y_All;        % (N x 1)
SVM_Data.ROI = ROI_Labels;

save(fullfile(OutputFolder,'Alzheimer_HonPearson_ML.mat'),'SVM_Data');

fprintf(' Classical ML (HON-Pearson) data saved\n');
