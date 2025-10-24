% batch_run_delta_c_experiments 批量运行Delta_c扫描实验
%
% 功能描述:
%   批量调用 run_delta_c_scan_with_params 函数，执行多组不同参数配置的实验。
%   每组实验结果独立保存，便于后续对比分析。
%
% 配置方法:
%   在下方的实验配置矩阵中定义所有要运行的实验参数组合
%
% 使用示例:
%   直接运行此脚本即可
%
% 作者: Claude Code
% 日期: 2025
% 版本: MATLAB 2025a兼容

clc;
clear;
close all;

fprintf('=======================================================\n');
fprintf('       Delta_c 批量实验系统\n');
fprintf('=======================================================\n\n');

%% ===================== 实验配置 =====================
% 每一行定义一组实验参数：[重复次数, 群体数量, 噪声强度, 并行核心数]
%
% 参数说明:
%   列1: num_runs        - 每个参数点的重复次数 (建议: 30-100)
%   列2: N               - 粒子群体数量 (例如: 100, 200, 300)
%   列3: noise_intensity - 角度噪声强度 (例如: 0.05, 0.10, 0.15)
%   列4: num_workers     - 并行核心数 (根据你的CPU核心数设置)

experiment_configs = [
    % num_runs,  N,    noise,   workers
    100,         200,   0,    200;        % 实验1: 200粒子，无噪声
    100,         200,   0,    200;        % 实验2: 200粒子，无噪声
    100,         200,   0,    200;        % 实验3: 200粒子，无噪声
    100,        400,   0,    200;        % 实验4: 400粒子，无噪声
    100,         400,   0,    200;        % 实验5: 400粒子，无噪声
    100,         400,   0,    200;        % 实验6: 400粒子，无噪声
    100,       800,   0,    200;        % 实验7: 800粒子，无噪声
    100,         800,   0,    200;        % 实验8: 800粒子，无噪声
    100,        800,   0,    200;        % 实验9: 800粒子，无噪声

];

% 为每组实验添加描述标签（可选，用于日志显示）
experiment_labels = {
    '200粒子，无噪声 - 实验1';
    '200粒子，无噪声 - 实验2';
    '200粒子，无噪声 - 实验3';
    '400粒子，无噪声 - 实验4';
    '400粒子，无噪声 - 实验5';
    '400粒子，无噪声 - 实验6';
    '800粒子，无噪声 - 实验7';
    '800粒子，无噪声 - 实验8';
    '800粒子，无噪声 - 实验9';
};

%% ===================== 批量实验信息显示 =====================
num_experiments = size(experiment_configs, 1);

fprintf('共配置 %d 组实验\n\n', num_experiments);
fprintf('实验配置列表:\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('序号  重复次数  群体数量  噪声强度  并行核心  描述\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
for i = 1:num_experiments
    fprintf(' %2d     %4d      %4d     %.3f      %3d      %s\n', ...
        i, ...
        experiment_configs(i, 1), ...  % num_runs
        experiment_configs(i, 2), ...  % N
        experiment_configs(i, 3), ...  % noise
        experiment_configs(i, 4), ...  % workers
        experiment_labels{i});
end
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');


%% ===================== 执行批量实验 =====================
fprintf('\n');
fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('开始批量实验执行...\n');
fprintf('═══════════════════════════════════════════════════════════\n\n');

batch_start_time = tic;

% 记录每组实验的结果路径和状态
results_summary = cell(num_experiments, 1);
results_paths = cell(num_experiments, 1);

% 依次执行每组实验
for exp_idx = 1:num_experiments
    fprintf('\n');
    fprintf('┌─────────────────────────────────────────────────────────┐\n');
    fprintf('│ 正在执行第 %d/%d 组实验                                   │\n', exp_idx, num_experiments);
    fprintf('│ 配置: %-48s │\n', experiment_labels{exp_idx});
    fprintf('└─────────────────────────────────────────────────────────┘\n\n');

    % 提取当前实验参数
    num_runs = experiment_configs(exp_idx, 1);
    N = experiment_configs(exp_idx, 2);
    noise = experiment_configs(exp_idx, 3);
    num_workers = experiment_configs(exp_idx, 4);

    % 执行实验
    try
        % 调用带参数的Delta_c扫描函数
        result = run_delta_c_scan_with_params(num_runs, N, noise, num_workers);

        % 记录成功信息
        results_paths{exp_idx} = result.timestamp;
        results_summary{exp_idx} = struct(...
            'status', 'success', ...
            'timestamp', result.timestamp, ...
            'max_delta_c', max(result.delta_c), ...
            'success_rate_c1', 1 - sum(result.error_count(:,1))/(num_runs*79), ...
            'success_rate_c2', 1 - sum(result.error_count(:,2))/(num_runs*79), ...
            'time_hours', result.total_time_hours);

        fprintf('\n✓ 第 %d/%d 组实验成功完成！\n', exp_idx, num_experiments);

    catch ME
        % 记录失败信息
        warning('第 %d 组实验执行失败: %s', exp_idx, ME.message);
        results_summary{exp_idx} = struct(...
            'status', 'failed', ...
            'error', ME.message);

        fprintf('\n✗ 第 %d/%d 组实验失败。\n', exp_idx, num_experiments);
    end

    % 显示进度和时间估计
    elapsed_time = toc(batch_start_time);
    avg_time_per_exp = elapsed_time / exp_idx;
    remaining_time = avg_time_per_exp * (num_experiments - exp_idx);

    fprintf('\n');
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('批量进度: %d/%d (%.1f%%)\n', exp_idx, num_experiments, ...
        100 * exp_idx / num_experiments);
    fprintf('已用时间: %.1f 小时 (%.0f 分钟)\n', ...
        elapsed_time / 3600, elapsed_time / 60);
    if exp_idx < num_experiments
        fprintf('预计剩余: %.1f 小时 (%.0f 分钟)\n', ...
            remaining_time / 3600, remaining_time / 60);
        fprintf('预计完成: %s\n', ...
            char(datetime('now') + seconds(remaining_time)));
    end
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
end

total_elapsed_time = toc(batch_start_time);

%% ===================== 批量实验总结 =====================
fprintf('\n\n');
fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('         批量实验全部完成！\n');
fprintf('═══════════════════════════════════════════════════════════\n\n');

fprintf('总耗时: %.2f 小时 (%.0f 分钟)\n', ...
    total_elapsed_time / 3600, total_elapsed_time / 60);
fprintf('完成时间: %s\n\n', char(datetime('now')));

% 统计成功/失败数
num_success = sum(cellfun(@(x) strcmp(x.status, 'success'), results_summary));
num_failed = num_experiments - num_success;

fprintf('实验成功数: %d/%d\n', num_success, num_experiments);
if num_failed > 0
    fprintf('实验失败数: %d\n', num_failed);
end

fprintf('\n详细结果汇总:\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
for i = 1:num_experiments
    fprintf('\n[实验 %d] %s\n', i, experiment_labels{i});
    fprintf('  参数: N=%d, noise=%.3f, runs=%d, workers=%d\n', ...
        experiment_configs(i, 2), experiment_configs(i, 3), ...
        experiment_configs(i, 1), experiment_configs(i, 4));

    if strcmp(results_summary{i}.status, 'success')
        fprintf('  状态: ✓ 成功\n');
        fprintf('  时间戳: %s\n', results_summary{i}.timestamp);
        fprintf('  Δc峰值: %.4f\n', results_summary{i}.max_delta_c);
        fprintf('  成功率: c1=%.1f%%, c2=%.1f%%\n', ...
            results_summary{i}.success_rate_c1 * 100, ...
            results_summary{i}.success_rate_c2 * 100);
        fprintf('  耗时: %.2f 小时\n', results_summary{i}.time_hours);
    else
        fprintf('  状态: ✗ 失败\n');
        fprintf('  错误: %s\n', results_summary{i}.error);
    end
end
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

%% ===================== 保存批量实验元信息 =====================
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
summary_dir = fullfile(pwd, 'data', 'experiments', 'batch_summaries');
if ~isfolder(summary_dir)
    mkdir(summary_dir);
end

summary_file = fullfile(summary_dir, sprintf('batch_summary_%s.mat', timestamp));

batch_info = struct();
batch_info.timestamp = timestamp;
batch_info.date = datetime('now');
batch_info.num_experiments = num_experiments;
batch_info.configurations = experiment_configs;
batch_info.labels = experiment_labels;
batch_info.results_summary = results_summary;
batch_info.total_time_seconds = total_elapsed_time;
batch_info.total_time_hours = total_elapsed_time / 3600;
batch_info.num_success = num_success;
batch_info.num_failed = num_failed;

save(summary_file, 'batch_info');
fprintf('批量实验元信息已保存至:\n%s\n\n', summary_file);

fprintf('所有实验已完成！数据可在以下位置找到:\n');
fprintf('  data/experiments/delta_c_scan/\n\n');
