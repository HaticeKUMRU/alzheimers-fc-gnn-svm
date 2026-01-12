%% =========================================================
%  ALZHEIMER HON-Pearson VERİ HAZIRLAMA (SVM İÇİN)
%% =========================================================

clear; clc; close all;

%% =========================================================
% 1. AYARLAR
%% =========================================================

fprintf('🚀 SVM için HON-Pearson veri hazırlanıyor...\n');

AnaKlasor = '/Users/haticekumru/Desktop/Alzheimer_fnets';
CiktiKlasoru = fullfile(pwd, 'ALZHEIMER_SVM_HON_Pearson');
if ~exist(CiktiKlasoru,'dir'), mkdir(CiktiKlasoru); end

% ROI seçimi (410 → 400)
ROI_Indeksleri = [1:200, 211:410];
ROI_Sayisi = numel(ROI_Indeksleri);

% Parametreler
Esik_Orani = 0.26;
GuvenliAtanh = @(x) atanh(max(min(x,0.999999), -0.999999));

%% =========================================================
% 2. VERİ SETİ
%% =========================================================

Gruplar = { ...
    struct('isim','CN','yol','45_CN_yeni','etiket',0), ...
    struct('isim','AD','yol','45_AD_yeni','etiket',1) ...
};

X_SVM = [];
y_SVM = [];

%% =========================================================
% 3. ANA İŞLEME DÖNGÜSÜ
%% =========================================================

for g = 1:numel(Gruplar)

    GrupYolu = fullfile(AnaKlasor, Gruplar{g}.yol);
    Dosyalar = dir(fullfile(GrupYolu, '*.txt'));

    fprintf('➡️  %s grubu (%d dosya)\n', Gruplar{g}.isim, numel(Dosyalar));

    for i = 1:numel(Dosyalar)
        try
            %% --- A) VERİ OKUMA ---
            HamMatris = readmatrix(fullfile(GrupYolu, Dosyalar(i).name));
            Matris = HamMatris(ROI_Indeksleri, ROI_Indeksleri);

            if size(Matris,1) ~= ROI_Sayisi, continue; end
            Matris(~isfinite(Matris)) = 0;
            Matris(1:ROI_Sayisi+1:end) = 0;

            %% --- B) FISHER-Z (TEK KEZ) ---
            Matris_Z = GuvenliAtanh(Matris);

            %% --- C) HON (Spearman) ---
            HON = corr(Matris_Z, 'Type','Pearson','Rows','pairwise');
            HON(~isfinite(HON)) = 0;
            HON(1:ROI_Sayisi+1:end) = 0;

            %% --- D) EŞİKLEME ---
            ust = abs(HON(triu(true(ROI_Sayisi),1)));
            if ~isempty(ust)
                esik = prctile(ust,(1-Esik_Orani)*100);
                HON(abs(HON) < esik) = 0;
            end

            %% --- E) SVM ÖZNİTELİKLERİ ---
            % HON ÜZERİNDEN NODE STRENGTH
            NodeStrength = mean(abs(HON),2)';
            X_SVM = [X_SVM; NodeStrength];
            y_SVM = [y_SVM; Gruplar{g}.etiket];

        catch
            continue;
        end
    end
end

%% =========================================================
% 4. VERİ TEMİZLİĞİ
%% =========================================================

X_SVM(isnan(X_SVM)) = 0;
ROI_Etiketleri = compose("ROI_%03d",(1:ROI_Sayisi)');
X_MLP = X_SVM;   
y_Tum = y_SVM;   


%% =========================================================
% 5. SVM VERİ KAYDI
%% =========================================================

SVM_Veri.X   = X_MLP;        % (N x 400)
SVM_Veri.y   = y_Tum;        % (N x 1)
SVM_Veri.ROI = ROI_Etiketleri;

save(fullfile(CiktiKlasoru,'Alzheimer_HonPearson_ML.mat'),'SVM_Veri');

fprintf('✅ Classical ML (HON-Pearson) verisi kaydedildi\n');
