clear; clc; close all;

%% =========================================================
% 1. GENERAL SETTINGS
%% =========================================================

ProjectRoot = fileparts(mfilename('fullpath'));
MainFolder  = ProjectRoot;
OutputFolder = fullfile(ProjectRoot, 'ALZHEIMER_ML_VERISI_Spearman');
if ~exist(OutputFolder, 'dir'), mkdir(OutputFolder); end

% ROI selection (410 → 400)
ROI_Indices = [1:200, 211:410];
ROI_Count = numel(ROI_Indices);

% Connectivity parameters
Threshold_Ratio = 0.26;   % Keep strongest 26% of connections
SafeAtanh = @(x) atanh(max(min(x,0.999999),-0.999999));

%% =========================================================
% 2. DATASET DEFINITION
%% =========================================================

Groups = {'45_CN_yeni', '45_AD_yeni'};
Labels = [0, 1];   % 0: Cognitively Normal (CN), 1: Alzheimer’s Disease (AD)

X_All = [];          % (N x 400 x 400) → GNN 
X_MLP = [];          % (N x 400) → MLP
y_All = [];

fprintf(' Data processing started...\n');

%% =========================================================
% 3. MAIN DATA PROCESSING LOOP
%% =========================================================

for g = 1:numel(Groups)
    
    GroupPath = fullfile(MainFolder, Groups{g});
    Files = dir(fullfile(GroupPath, '*.txt'));
    
    fprintf('  %s group (%d files)\n', Groups{g}, numel(Files));
    
    for i = 1:numel(Files)
        try
            %% --- A) DATA READING ---
            FilePath = fullfile(GroupPath, Files(i).name);
            Signal = readmatrix(FilePath);

            % Ensure Time x ROI format
            if size(Signal,1) < size(Signal,2)
                Signal = Signal';
            end

            %% --- B) ROI CROPPING ---
            if size(Signal,2) < 410, continue; end
            Signal = Signal(:, ROI_Indices);

            %% --- C) SPEARMAN FC COMPUTATION ---
            FC = corr(Signal,'Type','Spearman');
            FC(isnan(FC)) = 0;
            FC(1:ROI_Count+1:end) = 0;  % Zero diagonal

            %% --- D) THRESHOLDING ---
            upper_triangle = abs(FC(triu(true(ROI_Count),1)));
            threshold = prctile(upper_triangle, (1-Threshold_Ratio)*100);
            FC(abs(FC) < threshold) = 0;

            %% --- E) FISHER-Z TRANSFORMATION ---
            FC_Z = SafeAtanh(FC);

            %% --- F) SAVE (GNN / DeepSet) ---
            X_All = cat(3, X_All, FC_Z);
            y_All = [y_All; Labels(g)];

            %% --- G) FEATURE EXTRACTION (MLP) ---
            NodeStrength = mean(abs(FC_Z),2)';
            X_MLP = [X_MLP; NodeStrength];

        catch
            continue;
        end
    end
end

%% =========================================================
% 4. DATA CLEANING
%% =========================================================

X_MLP(isnan(X_MLP)) = 0;
ROI_Labels = compose("ROI_%03d",1:ROI_Count)';

%% =========================================================
% 5. DATA SAVING (CLASSICAL ML)
%% =========================================================

SVM_Data.X   = X_MLP;        % (N x 400)
SVM_Data.y   = y_All;        % (N x 1)
SVM_Data.ROI = ROI_Labels;

save(fullfile(OutputFolder,'Alzheimer_Spearman_ML.mat'),'SVM_Data');

fprintf(' Classical ML (SVM) data saved\n');
