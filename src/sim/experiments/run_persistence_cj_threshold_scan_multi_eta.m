% run_persistence_cj_threshold_scan_multi_eta
% =========================================================================
% 功能：
%   - 复用 run_persistence_cj_threshold_scan，在多组噪声幅度 η 下批量生成结果
%   - 每个 η 值通过 η = sqrt(2 D_θ) 反推 angleNoiseIntensity，并保存对应的图与数据
%   - 方便与响应性扫描相比对，保持命名与目录结构一致
%
% 使用说明：
%   - 直接运行本脚本即可；如需额外噪声点，修改 eta_values 数组
% =========================================================================

clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir, '..', '..', '..')));

eta_values = [0.10, 0.15, 0.20, 0.25, 0.30];
num_levels = numel(eta_values);

single_scan_script = fullfile(script_dir, 'run_persistence_cj_threshold_scan.m');
if ~isfile(single_scan_script)
    error('未找到 run_persistence_cj_threshold_scan.m，无法执行批量扫描。');
end

fprintf('=================================================\n');
fprintf('  批量运行持久性扫描：η ∈ [%0.2f, %0.2f]\n', min(eta_values), max(eta_values));
fprintf('  总共 %d 个噪声水平\n', num_levels);
fprintf('=================================================\n\n');

for idx = 1:num_levels
    run_single_eta(single_scan_script, eta_values(idx), idx, num_levels);
end

fprintf('\n所有噪声幅度的持久性扫描已完成。\n');

%% -------------------------------------------------------------------------
function run_single_eta(single_scan_script, eta_value, idx, total_levels)
    angle_noise_override = (eta_value.^2) / 2;          % 根据 η 计算 angleNoiseIntensity
    persistence_scan_preserve_workspace = true;         % 告知目标脚本跳过 clear

    eta_tag = strrep(sprintf('eta_%0.3f', eta_value), '.', 'p');
    fprintf('>>> [%d/%d] 运行 η = %.3f (angleNoiseIntensity = %.6f)，标签 %s\n', ...
        idx, total_levels, eta_value, angle_noise_override, eta_tag);

    run(single_scan_script);

    if exist('timestamp', 'var')
        fprintf('    η = %.3f 扫描完成，时间戳 %s。\n\n', eta_value, timestamp);
    else
        fprintf('    η = %.3f 扫描完成。\n\n', eta_value);
    end

    clear angle_noise_override persistence_scan_preserve_workspace timestamp results ...
        P_raw P_norm_mean P_norm_std P_norm_sem D_raw
end
