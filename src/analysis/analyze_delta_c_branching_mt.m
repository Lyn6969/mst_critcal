% analyze_delta_c_branching_mt - 批量对比 Δc 峰值与分支比接近 1 时的阈值
%
% 功能简介：
%   扫描指定批次目录下所有实验结果 (data.mat)，计算：
%     1. Δc = c_2 - c_1 的最大值所对应的 M_T
%     2. M_T < 1 区间内，分支比最接近 1 的阈值
%   并输出两者是否一致的统计。
%
% 使用说明：
%   根据实际数据目录调整 BASE_REL_DIR，运行脚本即可在控制台查看结果。

clear; clc; close all;

%% -------------------- 基本配置 --------------------
% 相对项目根目录的批量数据目录
BASE_REL_DIR = fullfile('mst_critcal', 'data', 'experiments', ...
    'batch_delta_c_m_vs_1', '20251101_003447');
DATA_FILENAME = 'data.mat';      % 每个实验目录中的结果文件名
TARGET_PULSE = 1;                % 需要分析的初发个体数量（默认 m = 1）
MT_MATCH_TOLERANCE = 1e-6;       % 判断两个 M_T 是否相等的容差

%% -------------------- 路径解析 --------------------
script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(fileparts(script_dir)));
base_dir = fullfile(project_root, BASE_REL_DIR);

if ~isfolder(base_dir)
    error('目标目录不存在：%s', base_dir);
end

sub_entries = dir(base_dir);
sub_entries = sub_entries([sub_entries.isdir]);
sub_entries = sub_entries(~ismember({sub_entries.name}, {'.', '..'}));

if isempty(sub_entries)
    warning('目录 %s 下未找到任何实验子目录。', base_dir);
end

%% -------------------- 结果存储 --------------------
record_names = {};
record_mt_delta = [];
record_mt_branch = [];
record_diff = [];

%% -------------------- 主循环 --------------------
for idx = 1:numel(sub_entries)
    sub_name = sub_entries(idx).name;
    data_path = fullfile(sub_entries(idx).folder, sub_name, DATA_FILENAME);

    if ~isfile(data_path)
        fprintf('跳过目录（缺少 %s）：%s\n', DATA_FILENAME, data_path);
        continue;
    end

    data_struct = load(data_path, 'results');
    if ~isfield(data_struct, 'results')
        warning('文件 %s 中找不到 results 结构，已跳过。', data_path);
        continue;
    end

    results = data_struct.results;

    % --------- 计算 Δc 峰值对应的阈值 ---------
    if ~isfield(results, 'c_mean')
        warning('结果缺少 c_mean 字段，已跳过：%s', data_path);
        continue;
    end
    if size(results.c_mean, 2) < 2
        warning('结果中 c_mean 列数不足，无法计算 Δc：%s', data_path);
        continue;
    end
    thresholds = results.cj_thresholds(:);
    c1_mean = results.c_mean(:, 1);
    c2_mean = results.c_mean(:, 2);
    delta_c = c2_mean - c1_mean;
    [~, max_idx] = max(delta_c);
    mt_delta = thresholds(max_idx);

    % --------- 计算分支比接近 1 的阈值 ---------
    if ~isfield(results, 'branching_mean')
        warning('结果缺少 branching_mean 字段，已跳过：%s', data_path);
        continue;
    end
    pulse_counts = results.pulse_counts(:);
    target_col = find(pulse_counts == TARGET_PULSE, 1);
    if isempty(target_col)
        warning('未找到 pulse = %d 的分支比数据：%s', TARGET_PULSE, data_path);
        continue;
    end
    branching_mean = results.branching_mean(:, target_col);
    mask_lt_one = thresholds < 1;
    if any(mask_lt_one)
        thresholds_lt = thresholds(mask_lt_one);
        branching_lt = branching_mean(mask_lt_one);
        [~, closest_idx] = min(abs(branching_lt - 1));
        mt_branch = thresholds_lt(closest_idx);
    else
        mt_branch = NaN;
    end

    % --------- 汇总结果 ---------
    record_names{end + 1, 1} = sub_name; %#ok<AGROW>
    record_mt_delta(end + 1, 1) = mt_delta; %#ok<AGROW>
    record_mt_branch(end + 1, 1) = mt_branch; %#ok<AGROW>
    record_diff(end + 1, 1) = mt_branch - mt_delta; %#ok<AGROW>
end

%% -------------------- 输出总结 --------------------
if isempty(record_names)
    fprintf('没有可用的数据条目，脚本结束。\n');
    return;
end

fprintf('\n====== Δc 峰值与分支比接近 1 的阈值对比 ======\n');
fprintf('%-40s | %-12s | %-12s | %-12s | 匹配\n', ...
    '实验目录', 'M_T(Δc峰)', 'M_T(分支比)', '差值');
fprintf('%s\n', repmat('-', 1, 90));

match_indices = [];
for idx = 1:numel(record_names)
    mt_delta = record_mt_delta(idx);
    mt_branch = record_mt_branch(idx);
    diff_value = record_diff(idx);

    if isnan(mt_branch)
        match_flag = '无有效M_T<1';
    else
        if abs(diff_value) <= MT_MATCH_TOLERANCE
            match_flag = '是';
            match_indices(end + 1) = idx; %#ok<AGROW>
        else
            match_flag = '否';
        end
    end

    fprintf('%-40s | %12.4f | %12.4f | %12.6f | %s\n', ...
        record_names{idx}, mt_delta, mt_branch, diff_value, match_flag);
end

fprintf('%s\n', repmat('-', 1, 90));
if isempty(match_indices)
    fprintf('提示：没有发现 Δc 峰值阈值与分支比阈值完全一致的实验。\n');
else
    fprintf('总计 %d 个实验的两个阈值在容差 %.1e 内一致。\n', ...
        numel(match_indices), MT_MATCH_TOLERANCE);
end
