clc; clear; close all;

%%导入数据



[h, w, bands] = size(data);
X = reshape(data, [], bands);       
y = reshape(labels, [], 1);         

mask = y ~= 0;
X = X(mask, :);   
y = y(mask);      

X = zscore(X);

Y = dummyvar(y);  

[~, mu, sigma] = zscore(X);
X = (X - mu) ./ sigma;

[XL1, ~, XS1, ~, ~] = plsregress(X, Y, 50);
X_trt_part = XS1 * XL1';
X_residual = X - X_trt_part;


[XL2, ~, XS2, ~, ~] = plsregress(X_residual, yba_train, 100);
X_batch = XS2 * XL2';
X_clean =  X_train - X_batch;

X = X + X_residual;

maxLV = 50;
[~, ~, ~, ~, ~, PCTVAR, ~, ~] = plsregress(X, Y, maxLV);


cumVar = cumsum(100 * sum(PCTVAR, 1));
optLV = find(cumVar >= 95, 1);
fprintf('选取的最佳成分数: %d\n', optLV);


[XL, YL, XS, YS, beta, PCTVAR, MSE, stats] = plsregress(X, Y, optLV);


W0 = stats.W ./ sqrt(sum(stats.W.^2, 1));
p = size(XL, 1); 

sumSq = sum(XS.^2, 1) .* sum(YL.^2, 1);  
vipScore = sqrt(p * sum(sumSq .* (W0.^2), 2) ./ sum(sumSq, 2));

mainColor = [0.2, 0.4, 0.8];
highlightColor = [0.85, 0.33, 0.1];
thresholdColor = [0.2, 0.2, 0.2];


figure('Color', 'w');
scatter(1:length(vipScore), vipScore, 10, ...
    'MarkerEdgeColor', mainColor, ...
    'MarkerFaceColor', mainColor, ...
    'Marker', 'o'); hold on;

indVIP = find(vipScore >= 1);
scatter(indVIP, vipScore(indVIP), 20, ...
    'MarkerEdgeColor', highlightColor, ...
    'MarkerFaceColor', highlightColor, ...
    'Marker', 'p');


plot([1, length(vipScore)], [1, 1], '--', 'Color', thresholdColor, 'LineWidth', 1.2);


xlabel('预测变量 Index', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('VIP 得分', 'FontSize', 12, 'FontWeight', 'bold');
legend({'全部变量', 'VIP ≥ 1'}, 'Location', 'Best');
axis tight;
grid on;
box off;
set(gca, 'FontSize', 12);

[~, top10Indices] = sort(vipScore, 'descend');  
top10Indices = top10Indices(1:10);  


disp('前10个最重要的波段索引：');
disp(top10Indices);

% 可视化前10个最重要的波段
figure('Color', 'w');
for i = 1:10
    figure(i+1)
    band_image = reshape(data(:, :, top10Indices(i)), h, w);
    imagesc(band_image); colormap('gray'); axis image; colorbar;
    title(['波段 ', num2str(top10Indices(i))], 'FontSize', 12);
end
