% run_responsiveness_cj_threshold_scan_multi_eta
% =========================================================================
% 功能：
%   - 依次扫描多组噪声幅度 η，复用 run_responsiveness_cj_threshold_scan 脚本
%   - 每个 η 值都会转换为 angleNoiseIntensity，并调用单次扫描脚本生成图像/数据
%   - 便于批量生成与噪声相关的响应性曲线，保持文件命名一致
%
% 使用说明：
%   - 直接运行本脚本即可，内部会循环 5 个 η 值（0.10~0.30）
%   - 如需调整噪声列表，修改 eta_values 即可
% =========================================================================

clc; clear; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(script_dir, '..', '..', '..')));

eta_values = [0 0.05 0.10, 0.15, 0.20, 0.25, 0.30];   % 指定需要批量运行的噪声幅度
num_levels = numel(eta_values);

single_scan_script = fullfile(script_dir, 'run_responsiveness_cj_threshold_scan.m');
if ~isfile(single_scan_script)
    error('未找到 run_responsiveness_cj_threshold_scan.m，无法继续批量扫描。');
end

fprintf('=================================================\n');
fprintf('  批量运行响应性扫描：η ∈ [%0.2f, %0.2f]\n', min(eta_values), max(eta_values));
fprintf('  总共 %d 个噪声水平\n', num_levels);
fprintf('=================================================\n\n');

for idx = 1:num_levels
    run_single_eta(single_scan_script, eta_values(idx), idx, num_levels);
end

fprintf('\n所有噪声幅度的响应性扫描已完成。\n');

%% -------------------------------------------------------------------------
function run_single_eta(single_scan_script, eta_value, idx, total_levels)
    angle_noise_override = (eta_value.^2) / 2;   % 根据 η=√(2D) 反推 angleNoiseIntensity
    resp_scan_preserve_workspace = true;        % 提示目标脚本跳过 clear，保留覆盖参数

    eta_tag = strrep(sprintf('eta_%0.3f', eta_value), '.', 'p');
    fprintf('>>> [%d/%d] 运行 η = %.3f (angleNoiseIntensity = %.6f)，标签 %s\n', ...
        idx, total_levels, eta_value, angle_noise_override, eta_tag);

    run(single_scan_script);   % 复用单次扫描脚本

    if exist('timestamp', 'var')
        fprintf('    η = %.3f 扫描完成，时间戳 %s。\n\n', eta_value, timestamp);
    else
        fprintf('    η = %.3f 扫描完成。\n\n', eta_value);
    end
end
