% run_delta_c_m_vs_1_scan_multi_eta 批量运行 Delta_c (1 vs m) 扫描（多噪声幅度 η）
% =========================================================================
% 功能：
%   - 在固定群体规模 N=200 的前提下，分别在多组噪声幅度 η 下
%     调用 run_delta_c_m_vs_1_scan_parallel_with_N 进行 Delta_c (1 vs m) 扫描。
%   - 每个 η 值都会转换为 angleNoiseIntensity = η^2 / 2，并在输出目录 /
%     data.mat 文件名 / 快览图文件名中写入 eta 标签（例如 eta_0p200）。
%
% 使用说明：
%   - 直接运行本脚本即可；默认 η 列表为 [0.20, 0.25, 0.30]。
%   - 如需修改噪声水平，只需调整下方 eta_values 即可。
%   - 单次扫描的详细参数（cj 阈值范围、重复次数等）在
%     run_delta_c_m_vs_1_scan_parallel_with_N 中定义。
% =========================================================================

clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir, '..', '..', '..')));

% 固定群体规模
N_value = 200;

% 需要扫描的噪声幅度 η 列表
eta_values = [0.20, 0.25, 0.30];
num_levels = numel(eta_values);
repeat_times = 5;  % 每个噪声幅度重复运行的次数

fprintf('=================================================\n');
fprintf('  批量运行 Delta_c (1 vs m) 扫描\n');
fprintf('  N = %d\n', N_value);
fprintf('  噪声 η 列表: %s\n', mat2str(eta_values));
fprintf('=================================================\n\n');

results_all = cell(num_levels, repeat_times);
peak_summary(repeat_times * num_levels) = struct( ...
    'eta', NaN, 'repeat', NaN, 'peak_value', NaN, 'peak_threshold', NaN);
summary_idx = 0;

for idx = 1:num_levels
    eta_value = eta_values(idx);
    for rep = 1:repeat_times
        fprintf('>>> η = %.3f | 第 %d/%d 次运行 (%d/%d 噪声)\n', ...
            eta_value, rep, repeat_times, idx, num_levels);

        results = run_delta_c_m_vs_1_scan_parallel_with_N(N_value, eta_value);
        results_all{idx, rep} = results;

        [peak_value, peak_threshold] = compute_delta_peak_from_results(results);
        summary_idx = summary_idx + 1;
        peak_summary(summary_idx) = struct( ...
            'eta', eta_value, ...
            'repeat', rep, ...
            'peak_value', peak_value, ...
            'peak_threshold', peak_threshold);

        if isfield(results, 'timestamp')
            fprintf('    η = %.3f 第 %d 次完成，时间戳 %s。\n', ...
                eta_value, rep, results.timestamp);
        else
            fprintf('    η = %.3f 第 %d 次完成。\n', eta_value, rep);
        end

        if ~isnan(peak_value)
            fprintf('    Δc 峰值 = %.4f，对应阈值 M_T = %.4f。\n\n', ...
                peak_value, peak_threshold);
        else
            fprintf('    Δc 峰值计算失败（数据为空或均为 NaN）。\n\n');
        end
    end
end

fprintf('所有 η 水平的 Delta_c (1 vs m) 扫描已完成。\n');

fprintf('\nΔc 峰值汇总：\n');
for idx = 1:summary_idx
    info = peak_summary(idx);
    if isnan(info.peak_value)
        fprintf('  η = %.3f | 第 %d 次 → 无有效峰值。\n', info.eta, info.repeat);
    else
        fprintf('  η = %.3f | 第 %d 次 → 峰值 %.4f @ M_T = %.4f\n', ...
            info.eta, info.repeat, info.peak_value, info.peak_threshold);
    end
end

function [peak_value, peak_threshold] = compute_delta_peak_from_results(results)
%COMPUTE_DELTA_PEAK_FROM_RESULTS 从结果结构中计算 Δc 峰值及其阈值
    peak_value = NaN;
    peak_threshold = NaN;

    if ~isfield(results, 'delta_c') || isempty(results.delta_c)
        return;
    end
    if ~isfield(results, 'cj_thresholds') || isempty(results.cj_thresholds)
        return;
    end

    delta_matrix = results.delta_c;
    thresholds = results.cj_thresholds(:);
    if isempty(delta_matrix) || isempty(thresholds)
        return;
    end

    tmp = delta_matrix;
    tmp(~isfinite(tmp)) = -inf;
    [max_value, lin_idx] = max(tmp(:));
    if isinf(max_value)
        return;
    end

    [row_idx, ~] = ind2sub(size(tmp), lin_idx);
    if row_idx > numel(thresholds)
        return;
    end

    peak_value = max_value;
    peak_threshold = thresholds(row_idx);
end
