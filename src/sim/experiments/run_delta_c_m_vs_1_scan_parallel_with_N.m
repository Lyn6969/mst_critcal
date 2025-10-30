function results = run_delta_c_m_vs_1_scan_parallel_with_N(N_input)
%RUN_DELTA_C_M_VS_1_SCAN_PARALLEL_WITH_N 支持自定义N值的并行扫描运动显著性阈值
%   基于原 run_delta_c_m_vs_1_scan_parallel 函数，增加对集群个体数目N的自定义支持
%
%   输入参数:
%   - N_input: 集群个体数目（如200, 400, 800等）
%
%   输出:
%   - results: 实验结果结构体，包含所有统计数据和元信息

%% 1. 实验设定
fprintf('=================================================\n');
fprintf('   Delta_c (1 vs m) 参数扫描实验 (N=%d)\n', N_input);
fprintf('=================================================\n\n');
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..')));
params = default_simulation_parameters();

% 使用传入的N值覆盖默认值
params.N = N_input;

cj_threshold_min = 0;
cj_threshold_max = 5.0;
cj_threshold_step = 0.1;
cj_thresholds = cj_threshold_min:cj_threshold_step:cj_threshold_max;
num_params = numel(cj_thresholds);

num_runs = 50;
m_values = 2:10;                        % 与 c1 对比的初发个体数量
pulse_counts = [1, m_values];
num_pulses = numel(pulse_counts);

trials_per_param = num_runs * num_pulses;
completed_tasks = 0;
total_tasks = num_params * trials_per_param;

fprintf('集群个体数目: %d\n', N_input);
fprintf('参数扫描范围: [%.1f, %.1f], 步长 %.1f，共 %d 个参数点\n', ...
    cj_threshold_min, cj_threshold_max, cj_threshold_step, num_params);
fprintf('初发个体数: %s (共 %d 种)\n', mat2str(pulse_counts), num_pulses);
fprintf('每个参数重复次数: %d，总实验次数: %d\n', num_runs, total_tasks);

%% 2. 并行池
pool = configure_parallel_pool();
fprintf('使用并行池: %d workers\n', pool.NumWorkers);

%% 3. 预分配
c_raw = NaN(num_params, num_runs, num_pulses);

experiment_start_time = tic;
progress_update_timer = tic;
progress_step = max(1, floor(total_tasks / 100));

timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_dir = ensure_data_directory_with_N(timestamp, N_input);
output_filename = fullfile(output_dir, 'data.mat');
temp_filename = fullfile(output_dir, 'temp.mat');

progress_queue = parallel.pool.DataQueue;
afterEach(progress_queue, @(inc) update_progress(inc));

%% 4. 并行仿真
fprintf('\n开始并行实验...\n');
fprintf('----------------------------------------\n');

shared_params = parallel.pool.Constant(params);
shared_pulse_counts = parallel.pool.Constant(pulse_counts);

parfor param_idx = 1:num_params
    local_c = NaN(num_runs, num_pulses);
    current_params = shared_params.Value;
    current_params.cj_threshold = cj_thresholds(param_idx);
    local_pulse_counts = shared_pulse_counts.Value;

    for run_idx = 1:num_runs
        for pulse_idx = 1:num_pulses
            pulse = local_pulse_counts(pulse_idx);
            local_c(run_idx, pulse_idx) = run_single_experiment(current_params, pulse);
            send(progress_queue, 1);
        end
    end

    c_raw(param_idx, :, :) = local_c;
end

total_elapsed_seconds = toc(experiment_start_time);
fprintf('\n实验执行完毕，用时 %.2f 分钟。\n', total_elapsed_seconds / 60);

%% 5. 统计
[c_mean, c_std, c_sem] = compute_statistics_3d(c_raw);
error_count = sum(isnan(c_raw), 2);
error_count = reshape(error_count, num_params, num_pulses);
success_rate_matrix = 1 - error_count / num_runs;
success_rate_per_pulse = 1 - sum(error_count, 1) / (num_runs * num_params);

c1_mean = c_mean(:, 1);
c1_sem = c_sem(:, 1);

delta_c = c_mean(:, 2:end) - c1_mean;
delta_sem = sqrt(c_sem(:, 2:end).^2 + c1_sem.^2);

%% 6. 保存
results = struct();
results.description = sprintf('Delta_c parameter scan (1 vs m) with N=%d', N_input);
results.parameters = params;
results.scan_variable = 'cj_threshold';
results.cj_thresholds = cj_thresholds;
results.num_runs = num_runs;
results.pulse_counts = pulse_counts;
results.c_raw = c_raw;
results.c_mean = c_mean;
results.c_std = c_std;
results.c_sem = c_sem;
results.delta_c = delta_c;
results.delta_sem = delta_sem;
results.timestamp = timestamp;
results.date = datetime('now');
results.total_experiments = total_tasks;
results.total_time_seconds = total_elapsed_seconds;
results.total_time_hours = total_elapsed_seconds / 3600;
results.error_count = error_count;
results.success_rate_matrix = success_rate_matrix;
results.success_rate_per_pulse = success_rate_per_pulse;
results.parallel_workers = pool.NumWorkers;
results.matlab_version = version;

save(output_filename, 'results', '-v7.3');
fprintf('结果已保存至: %s\n', output_filename);

if exist(temp_filename, 'file')
    delete(temp_filename);
end

%% 7. 摘要输出与可视化
fprintf('\n=================================================\n');
fprintf('                实验完成摘要\n');
fprintf('=================================================\n');
fprintf('集群个体数目: %d\n', N_input);
fprintf('总耗时: %.2f 小时\n', total_elapsed_seconds / 3600);
fprintf('成功率范围: %.1f%% ~ %.1f%% (各脉冲)\n', ...
    100 * min(success_rate_per_pulse), 100 * max(success_rate_per_pulse));

[max_delta, linear_idx] = max(delta_c(:));
[max_param_idx, max_m_idx] = ind2sub(size(delta_c), linear_idx);
optimal_cj = cj_thresholds(max_param_idx);
optimal_m = pulse_counts(max_m_idx + 1);
fprintf('Δc 峰值: %.4f (cj_threshold = %.2f, m = %d)\n', ...
    max_delta, optimal_cj, optimal_m);

quicklook_path = fullfile(output_dir, 'result');
render_quicklook_figure_with_N(cj_thresholds, pulse_counts, c_mean, c_sem, ...
    delta_c, delta_sem, error_count, success_rate_matrix, num_runs, pool.NumWorkers, N_input, quicklook_path);

fprintf('\n实验完成！\n');

%% 嵌套函数 ---------------------------------------------------------------
    function update_progress(increment)
        completed_tasks = completed_tasks + increment;
        if completed_tasks == 0
            return;
        end
        if toc(progress_update_timer) < 5 && ...
                mod(completed_tasks, progress_step) ~= 0
            return;
        end

        elapsed_seconds = toc(experiment_start_time);
        avg_time_per_task = elapsed_seconds / completed_tasks;
        remaining_seconds = avg_time_per_task * (total_tasks - completed_tasks);

        fprintf('  进度: %.1f%% (%d/%d) | 已用: %.1f分 | 预计剩余: %.1f分\n', ...
            100 * completed_tasks / total_tasks, completed_tasks, total_tasks, ...
            elapsed_seconds / 60, remaining_seconds / 60);

        progress_update_timer = tic;

        if mod(completed_tasks, progress_step * 10) == 0
            interim_results = struct('c_raw', c_raw, ...
                'timestamp', timestamp, 'completed_tasks', completed_tasks);
            save(temp_filename, 'interim_results', '-v7.3');
        end
    end
end

%% 辅助函数 ---------------------------------------------------------------
function result = run_single_experiment(params, pulse_count)
    try
        sim = ParticleSimulationWithExternalPulse(params);
        sim.external_pulse_count = pulse_count;
        cascade_size = sim.runSingleExperiment(pulse_count);
        result = cascade_size;
    catch ME
        warning('run_single_experiment:failure', ...
            'Experiment failed (N=%d, cj=%.3f, pulse=%d): %s', ...
            params.N, params.cj_threshold, pulse_count, ME.message);
        result = NaN;
    end
end

function render_quicklook_figure_with_N(thresholds, pulse_counts, c_mean, c_sem, ...
    delta_c, delta_sem, error_count, success_rate_matrix, num_runs, num_workers, N_input, output_prefix)
%RENDER_QUICKLOOK_FIGURE_WITH_N 生成包含N值信息的Delta_c扫描预览图

    fig = figure('Name', sprintf('Delta_c (1 vs m) 扫描结果 N=%d', N_input), ...
        'Position', [100, 100, 1350, 450]);

    % c1 与部分 c_m
    subplot(1, 3, 1);
    hold on;
    errorbar(thresholds, c_mean(:, 1), c_sem(:, 1), 'k-', 'LineWidth', 1.6, ...
        'DisplayName', 'c_1');
    colors = lines(numel(pulse_counts) - 1);
    for idx = 2:numel(pulse_counts)
        if idx > 6 && mod(idx, 2) == 1
            continue; % 适度稀疏曲线避免过拥挤
        end
        errorbar(thresholds, c_mean(:, idx), c_sem(:, idx), '-', ...
            'Color', colors(idx-1, :), 'LineWidth', 1.0, ...
            'DisplayName', sprintf('c_{%d}', pulse_counts(idx)));
    end
    xlabel('c_j threshold');
    ylabel('平均级联规模');
    title('级联规模 vs 阈值');
    legend('Location', 'northeastoutside');
    grid on;
    hold off;

    % Δc 曲线
    subplot(1, 3, 2);
    hold on;
    for idx = 1:size(delta_c, 2)
        errorbar(thresholds, delta_c(:, idx), delta_sem(:, idx), '-', ...
            'LineWidth', 1.2, ...
            'DisplayName', sprintf('\\Delta c_{%d-1}', pulse_counts(idx+1)));
    end
    xlabel('c_j threshold');
    ylabel('\Delta c = c_m - c_1');
    title('集群敏感性 (1 vs m)');
    legend('Location', 'northeastoutside');
    grid on;
    hold off;

    % 失败次数
    subplot(1, 3, 3);
    bar(thresholds, error_count, 'stacked');
    xlabel('c_j threshold');
    ylabel('失败次数');
    title(sprintf('失败统计 (每阈值，%d次 × %d脉冲)', num_runs, numel(pulse_counts)));
    grid on;

    sgtitle(sprintf('Delta_c (1 vs m) 参数扫描 (N=%d, %d次重复, %d workers)', ...
        N_input, num_runs, num_workers));

    savefig([output_prefix, '.fig']);
    print([output_prefix, '.png'], '-dpng', '-r300');
    close(fig);
end

function params = default_simulation_parameters()
    params.N = 200;  % 默认值，会被函数参数覆盖
    params.rho = 1;
    params.v0 = 1;
    params.angleUpdateParameter = 10;
    params.angleNoiseIntensity = 0;
    params.T_max = 400;
    params.dt = 0.1;
    params.radius = 5;
    params.deac_threshold = 0.1745;
    params.cj_threshold = 1;
    params.fieldSize = 50;
    params.initDirection = pi / 4;
    params.useFixedField = true;
    params.stabilization_steps = 100;
    params.forced_turn_duration = 200;
end

function [mean_values, std_values, sem_values] = compute_statistics_3d(raw_data)
    mean_values = NaN(size(raw_data, 1), size(raw_data, 3));
    std_values = mean_values;
    sem_values = mean_values;

    for param_idx = 1:size(raw_data, 1)
        for pulse_idx = 1:size(raw_data, 3)
            samples = squeeze(raw_data(param_idx, :, pulse_idx));
            samples = samples(~isnan(samples));
            if isempty(samples)
                continue;
            end
            mean_values(param_idx, pulse_idx) = mean(samples);
            std_values(param_idx, pulse_idx) = std(samples, 0);
            sem_values(param_idx, pulse_idx) = ...
                std_values(param_idx, pulse_idx) / sqrt(numel(samples));
        end
    end
end

function output_dir = ensure_data_directory_with_N(timestamp, N_input)
    base_dir = fullfile(pwd, 'data', 'experiments', 'delta_c_m_vs_1_scan');
    if ~isfolder(base_dir)
        mkdir(base_dir);
    end
    output_dir = fullfile(base_dir, sprintf('N%d_%s', N_input, timestamp));
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
end

function pool = configure_parallel_pool()
    if ~license('test', 'Distrib_Computing_Toolbox')
        error('需要 Parallel Computing Toolbox 才能运行此脚本。');
    end

    pool = gcp('nocreate');
    if isempty(pool)
        cluster = parcluster('local');
        max_workers = min(cluster.NumWorkers, 40);
        if max_workers < 1
            error('没有可用的并行工作线程。');
        end
        pool = parpool(cluster, max_workers);
    end
end