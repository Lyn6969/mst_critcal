% 批量 Delta_c (1 vs m) 实验脚本（简化版）
% ------------------------------------------------------------
% 按照指定的 N 列表依次调用 run_delta_c_m_vs_1_scan_parallel_with_N。
% 每个规模重复 num_repeats 次，结果目录会被移动到批处理专用目录。

clc;
clear;
close all;
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..')));
%% 参数配置
N_values = [200, 400, 800];
num_repeats = 5;
total_experiments = numel(N_values) * num_repeats;

script_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(fileparts(fileparts(script_dir)));
addpath(genpath(project_root));

%% 批处理目录
batch_timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
batch_dir = fullfile(project_root, 'data', 'experiments', ...
    'batch_delta_c_m_vs_1', batch_timestamp);
if ~isfolder(batch_dir)
    mkdir(batch_dir);
end
fprintf('批量实验目录: %s\n\n', batch_dir);

%% 批量运行
overall_timer = tic;
experiment_count = 0;

for N = N_values
    fprintf('=== 运行 N = %d 的实验系列 ===\n', N);
    for repeat_idx = 1:num_repeats
        experiment_count = experiment_count + 1;
        fprintf('  -> N = %d, 重复 %d/%d\n', N, repeat_idx, num_repeats);

        results = run_delta_c_m_vs_1_scan_parallel_with_N(N);

        source_dir = fullfile(project_root, 'data', 'experiments', ...
            'delta_c_m_vs_1_scan', sprintf('N%d_%s', N, results.timestamp));
        target_dir = fullfile(batch_dir, sprintf('N%d_run%d_%s', ...
            N, repeat_idx, results.timestamp));

        if isfolder(source_dir)
            movefile(source_dir, target_dir);
            fprintf('     结果目录 → %s\n', target_dir);
        else
            fprintf('     警告: 结果目录未找到 (%s)\n', source_dir);
        end

        fprintf('     Max Δc = %.4f\n', max(results.delta_c(:)));

        elapsed = toc(overall_timer);
        fprintf('     进度: %d/%d (%.1f%%)，已用 %.2f 小时\n', ...
            experiment_count, total_experiments, ...
            100 * experiment_count / total_experiments, elapsed / 3600);
    end
    fprintf('\n');
end

fprintf('批量实验完成！所有结果已整理至 %s\n', batch_dir);
