% plot_persistence_cj_threshold_scan_multi_eta
% 功能：在同一张图中叠加多组噪声幅度的持久性曲线，便于横向对比。

clear; clc; close all;

%% 基本配置 -------------------------------------------------------------
eta_values = [0.10, 0.20, 0.30];
result_files = { ...
    'persistence_cj_scan_20251111_203019_eta_0p100.mat', ...
    'persistence_cj_scan_20251111_203325_eta_0p200.mat', ...
    'persistence_cj_scan_20251111_203538_eta_0p300.mat', ...
};

FIG_WIDTH = 400;
FIG_HEIGHT = 200;
BASE_COLOR = [38, 94, 180] / 255;
LINE_WIDTH = 2.5;
AXIS_LINE_WIDTH = 1.5;
FONT_NAME = 'Arial';
LABEL_FONT_SIZE = 12;
LABEL_FONT_WEIGHT = 'Bold';
TICK_FONT_SIZE = 13;
TICK_FONT_WEIGHT = 'Bold';

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));
results_dir = fullfile(project_root, 'results', 'persistence');
if ~isfolder(results_dir)
    error('未找到结果目录: %s', results_dir);
end

pic_dir = fullfile(project_root, 'pic');
if ~exist(pic_dir, 'dir')
    mkdir(pic_dir);
end

if numel(result_files) ~= numel(eta_values)
    error('result_files 与 eta_values 数量不一致，请补齐文件列表。');
end

%% 读取数据 -------------------------------------------------------------
curves = struct('eta', {}, 'cj', {}, 'P', {});
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
    if ~isfield(results, 'cj_thresholds') || ~isfield(results, 'P_mean')
        warning('文件 %s 缺少必要字段，已跳过。', file_name);
        continue;
    end
    curves(end+1).eta = eta; %#ok<SAGROW>
    curves(end).cj = results.cj_thresholds(:);
    curves(end).P = results.P_mean(:);
end

if isempty(curves)
    error('未加载到任何有效数据，请确认文件名。');
end

baseline_cj = curves(1).cj;
for k = 2:numel(curves)
    if numel(curves(k).cj) ~= numel(baseline_cj) || any(abs(curves(k).cj - baseline_cj) > 1e-8)
        error('不同 η 的 cj 网格不一致，无法叠加绘图。');
    end
end

%% 绘图 ---------------------------------------------------------------
fig = figure('Position', [150, 150, FIG_WIDTH, FIG_HEIGHT], 'Color', 'white');
ax = axes('Parent', fig);
hold(ax, 'on');

solid_levels = sum([curves.eta] > 0);
shade_levels = linspace(0, 0.55, max(solid_levels, 1));
solid_idx = 1;
legend_entries = strings(1, numel(curves));

for i = 1:numel(curves)
    eta = curves(i).eta;
    cj = curves(i).cj;
    P = curves(i).P;
    if eta == 0
        color = BASE_COLOR;
        style = '--';
    else
        shade = shade_levels(min(solid_idx, numel(shade_levels)));
        color = BASE_COLOR + shade * (1 - BASE_COLOR);
        color = min(color, 1);
        style = '-';
        solid_idx = solid_idx + 1;
    end
    plot(ax, cj, P, style, 'LineWidth', LINE_WIDTH, 'Color', color);
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
ylabel(ax, 'Persistence', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE, 'FontWeight', LABEL_FONT_WEIGHT);
xlim(ax, [min(baseline_cj), max(baseline_cj)]);

P_cells = arrayfun(@(c) c.P, curves, 'UniformOutput', false);
max_val = max(cellfun(@(v) max(v), P_cells));
ylim(ax, [0, max_val * 1.05]);

legend(ax, legend_entries, 'Location', 'southwest', 'Box', 'off');

%% 导出 ---------------------------------------------------------------
output_path = fullfile(pic_dir, 'persistence_vs_mt_multi_eta.pdf');
exportgraphics(fig, output_path, 'ContentType', 'vector');
fprintf('多噪声持久性图已输出至: %s\n', output_path);
