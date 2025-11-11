% plot_responsiveness_cj_threshold_scan_multi_eta
% 功能：汇集多组噪声幅度 η 的 cj 阈值扫描结果，并在同一张图中绘制响应性曲线。

clear; clc; close all;

%% 基本配置 -------------------------------------------------------------
eta_values = [0.0, 0.10, 0.20, 0.30];
result_files = { ...
    'responsiveness_cj_scan_20251111_173212_eta_0p000.mat', ...
    'responsiveness_cj_scan_20251111_173429_eta_0p100.mat', ...
    'responsiveness_cj_scan_20251111_173647_eta_0p200.mat', ...
    'responsiveness_cj_scan_20251111_173907_eta_0p300.mat' ...
};
FIG_WIDTH = 400;
FIG_HEIGHT = 200;
BASE_RED = [0.85, 0.1, 0.1];
LINE_WIDTH = 2.5;
AXIS_LINE_WIDTH = 1.5;
FONT_NAME = 'Arial';
LABEL_FONT_SIZE = 12;
LABEL_FONT_WEIGHT = 'Bold';
TICK_FONT_SIZE = 13;
TICK_FONT_WEIGHT = 'Bold';

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
results_dir = fullfile(project_root, 'results', 'responsiveness');
if ~isfolder(results_dir)
    error('未找到结果目录: %s', results_dir);
end

pic_dir = fullfile(project_root, 'pic');
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end

if numel(result_files) ~= numel(eta_values)
    error('result_files 与 eta_values 长度不一致，请补全文件名列表。');
end

%% 读取数据 -------------------------------------------------------------
curves = struct('eta', {}, 'cj', {}, 'R', {});
for idx = 1:numel(eta_values)
    eta = eta_values(idx);
    file_name = strtrim(result_files{idx});
    if isempty(file_name)
        warning('η=%.3f 未提供文件名，已跳过。', eta);
        continue;
    end
    mat_path = fullfile(results_dir, file_name);
    if ~isfile(mat_path)
        warning('η=%.3f 的文件不存在: %s，已跳过。', eta, mat_path);
        continue;
    end
    data = load(mat_path, 'results');
    results = data.results;
    curves(end+1).eta = eta; %#ok<SAGROW>
    curves(end).cj = results.cj_thresholds(:);
    curves(end).R = results.R_mean(:);
end

if isempty(curves)
    error('未加载到任何有效数据，请确认结果文件是否存在。');
end

% 统一阈值向量，用于坐标设置
all_cj = curves(1).cj;
for c = 2:numel(curves)
    if numel(curves(c).cj) ~= numel(all_cj) || any(abs(curves(c).cj - all_cj) > 1e-8)
        error('不同噪声结果的 cj 阈值网格不一致，无法叠加绘图。');
    end
end

%% 绘图 ---------------------------------------------------------------
fig = figure('Position', [160, 160, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
ax = axes('Parent', fig);
hold(ax, 'on');

solid_count = sum([curves.eta] > 0);
if solid_count > 0
    shade_levels = linspace(0, 0.55, solid_count);
else
    shade_levels = [];
end
solid_idx = 1;

legend_entries = strings(1, numel(curves));

for i = 1:numel(curves)
    eta = curves(i).eta;
    cj = curves(i).cj;
    R = curves(i).R;
    if eta == 0
        color = BASE_RED;
        style = '--';
    else
        shade = shade_levels(max(1, solid_idx));
        color = BASE_RED + shade * (1 - BASE_RED);
        color = min(color, 1);
        style = '-';
        solid_idx = solid_idx + 1;
    end
    plot(ax, cj, R, style, 'LineWidth', LINE_WIDTH, 'Color', color);
    legend_entries(i) = sprintf('\\eta = %.2f', eta);
end

ax.FontName = FONT_NAME;
ax.FontSize = TICK_FONT_SIZE;
ax.FontWeight = TICK_FONT_WEIGHT;
ax.LineWidth = AXIS_LINE_WIDTH;
ax.TickDir = 'in';
box(ax, 'on');
grid(ax, 'off');
xlabel(ax, 'M_T', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
ylabel(ax, 'Responsivity', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
xlim(ax, [min(all_cj), max(all_cj)]);
ylim(ax, [0, max(cellfun(@(r) max(r), {curves.R})) * 1.05]);

legend(ax, legend_entries, 'Location', 'southwest', 'Box', 'off');

%% 导出 ---------------------------------------------------------------
% output_path = fullfile(pic_dir, 'responsiveness_vs_mt_multi_eta.pdf');
% exportgraphics(fig, output_path, 'ContentType', 'vector');
% fprintf('多噪声响应性图已输出至: %s\n', output_path);
