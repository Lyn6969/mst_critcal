% analyze_batch_m_vs_mplus1_peak - 批量输出 Δc (m→m+1) 峰值及对应 M_T
%
% 功能：
%   扫描指定批次目录（每个子目录包含一个 data.mat），对每个实验结果：
%   1) 由 c_mean 计算 Δc_{m→m+1} = c_{m+1} - c_m
%   2) 在全部 m 与全部阈值上寻找最大值及其对应的 M_T
%   在控制台输出：
%     <实验子目录> : Δc (m vs m+1) 最大值 <数值> 对应的阈值 M_T = <Mt>
%
% 使用：
%   直接运行本脚本。若需更换批次路径，修改 BASE_BATCH_DIR_REL 常量。

clear; clc; close all;

%% -------------------- 配置 --------------------
% 相对项目根目录的批次目录
BASE_BATCH_DIR_REL = fullfile('data', 'experiments', 'batch_delta_c_m_vs_1', '20251103_104930');
DATA_FILENAME = 'data.mat';

%% -------------------- 路径解析 --------------------
% 本脚本位于项目根目录下的 src/analysis/ 目录，向上两级即为项目根
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(script_dir));

base_dir = fullfile(project_root, BASE_BATCH_DIR_REL);
if ~isfolder(base_dir)
    error('目标目录不存在：%s', base_dir);
end

sub_dirs = dir(base_dir);
sub_dirs = sub_dirs([sub_dirs.isdir]);
sub_dirs = sub_dirs(~ismember({sub_dirs.name}, {'.', '..'}));

if isempty(sub_dirs)
    fprintf('目录下没有实验子目录：%s\n', base_dir);
    return;
end

fprintf('批次目录：%s\n', base_dir);
fprintf('共发现 %d 个实验结果目录。\n\n', numel(sub_dirs));

%% -------------------- 主循环 --------------------
for k = 1:numel(sub_dirs)
    sub_name = sub_dirs(k).name;
    data_path = fullfile(base_dir, sub_name, DATA_FILENAME);

    if ~isfile(data_path)
        fprintf('[跳过] 子目录缺少 %s ：%s\n', DATA_FILENAME, fullfile(base_dir, sub_name));
        continue;
    end

    S = load(data_path, 'results');
    if ~isfield(S, 'results')
        fprintf('[跳过] 文件不包含 results 结构：%s\n', data_path);
        continue;
    end

    R = S.results;
    if ~isfield(R, 'c_mean') || isempty(R.c_mean)
        fprintf('[跳过] 结果中缺少 c_mean：%s\n', data_path);
        continue;
    end
    if size(R.c_mean, 2) < 2
        fprintf('[跳过] c_mean 列数不足（需要 >=2）：%s\n', data_path);
        continue;
    end
    if ~isfield(R, 'cj_thresholds') || isempty(R.cj_thresholds)
        fprintf('[跳过] 结果中缺少 cj_thresholds：%s\n', data_path);
        continue;
    end

    thresholds = R.cj_thresholds(:);            % [P×1]
    c_mean = R.c_mean;                           % [P×M]
    delta_c_mm1 = c_mean(:, 2:end) - c_mean(:, 1:end-1); % [P×(M-1)]

    % 处理 NaN：将 NaN 设为 -Inf，避免干扰最大值搜索
    tmp = delta_c_mm1;
    tmp(~isfinite(tmp)) = -inf;

    [peak_value, lin_idx] = max(tmp(:));
    if isinf(peak_value)
        fprintf('[跳过] 全为 NaN/无效数据：%s\n', data_path);
        continue;
    end

    [peak_row, ~] = ind2sub(size(tmp), lin_idx);
    peak_mt = thresholds(peak_row);

    fprintf('%-40s : Δc (m vs m+1) 最大值 %.4f 对应的阈值 M_T = %.4f\n', sub_name, peak_value, peak_mt);
end
