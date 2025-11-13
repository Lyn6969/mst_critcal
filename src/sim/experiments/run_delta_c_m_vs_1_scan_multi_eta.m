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

fprintf('=================================================\n');
fprintf('  批量运行 Delta_c (1 vs m) 扫描\n');
fprintf('  N = %d\n', N_value);
fprintf('  噪声 η 列表: %s\n', mat2str(eta_values));
fprintf('=================================================\n\n');

results_all = cell(num_levels, 1);

for idx = 1:num_levels
    eta_value = eta_values(idx);
    fprintf('>>> [%d/%d] 运行 η = %.3f\n', idx, num_levels, eta_value);

    % 调用带 η 支持的扫描函数
    results = run_delta_c_m_vs_1_scan_parallel_with_N(N_value, eta_value);
    results_all{idx} = results;

    if isfield(results, 'timestamp')
        fprintf('    η = %.3f 完成，时间戳 %s。\n\n', eta_value, results.timestamp);
    else
        fprintf('    η = %.3f 完成。\n\n', eta_value);
    end
end

fprintf('所有 η 水平的 Delta_c (1 vs m) 扫描已完成。\n');

