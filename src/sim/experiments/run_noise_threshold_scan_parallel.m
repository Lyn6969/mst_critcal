function run_noise_threshold_scan_parallel()
%RUN_NOISE_THRESHOLD_SCAN_PARALLEL
%   并行扫描“运动显著性阈值 c_j × 角度噪声强度 σ”，
%   计算级联规模 (c_m)、集群敏感性 (Δc)、平均分支比 (b) 等指标。

%% 1. 参数范围设定
fprintf('=================================================\n');
fprintf('   噪声 × 阈值 扫描实验 (并行化版本)\n');
fprintf('=================================================\n\n');

params = default_simulation_parameters();

cj_threshold_min = 0;
cj_threshold_max = 5.0;
cj_threshold_step = 0.1;
cj_thresholds = cj_threshold_min:cj_threshold_step:cj_threshold_max;
num_cj = numel(cj_thresholds);

noise_min = 0.0;
noise_max = 1.0;
noise_step = 0.05;
noise_levels = noise_min:noise_step:noise_max;
num_noise = numel(noise_levels);

num_runs = 30;            % 每个组合的重复次数
pulse_counts = [1, 2];    % 保持与原 Δc 代码一致 (c1, c2)

total_tasks = num_cj * num_noise * num_runs * numel(pulse_counts);
completed_tasks = 0;

fprintf('阈值范围: [%.1f, %.1f], 噪声范围: [%.2f, %.2f]\n', ...
    cj_threshold_min, cj_threshold_max, noise_min, noise_max);
fprintf('阈值步长: %.2f (共 %d 个), 噪声步长: %.2f (共 %d 个)\n', ...
    cj_threshold_step, num_cj, noise_step, num_noise);
fprintf('每个组合重复次数: %d，总实验次数: %d\n', num_runs, total_tasks);

%% 2. 并行池
pool = configure_parallel_pool();
fprintf('使用并行池: %d workers\n', pool.NumWorkers);

%% 3. 数据预分配
c_raw = NaN(num_cj, num_noise, num_runs, numel(pulse_counts));

experiment_start_time = tic;
progress_update_timer = tic;
progress_step = max(1, floor(total_tasks / 100));

timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_dir = ensure_data_directory(timestamp);
output_filename = fullfile(output_dir, 'data.mat');
temp_filename = fullfile(output_dir, 'temp.mat');

progress_queue = parallel.pool.DataQueue;
afterEach(progress_queue, @(inc) update_progress(inc));

%% 4. 并行扫描
fprintf('\n开始并行实验...\n');
fprintf('----------------------------------------\n');

shared_params = parallel.pool.Constant(params);

% 新增广播变量包装
shared_noise_levels = parallel.pool.Constant(noise_levels);
shared_cj_thresholds = parallel.pool.Constant(cj_thresholds);
shared_pulse_counts = parallel.pool.Constant(pulse_counts);

parfor cj_idx = 1:num_cj
    local_c = NaN(num_noise, num_runs, numel(pulse_counts));
    
    % 一次性获取所有广播变量值，减少通信开销
    current_noise_levels = shared_noise_levels.Value;
    current_cj_thresholds = shared_cj_thresholds.Value;
    current_pulse_counts = shared_pulse_counts.Value;

    for noise_idx = 1:num_noise
        current_params = shared_params.Value;
        current_params.cj_threshold = current_cj_thresholds(cj_idx);
        current_params.angleNoiseIntensity = current_noise_levels(noise_idx);

        for run_idx = 1:num_runs
            for pulse_idx = 1:numel(current_pulse_counts)
                pulse = current_pulse_counts(pulse_idx);
                local_c(noise_idx, run_idx, pulse_idx) = run_single_experiment(current_params, pulse);
                send(progress_queue, 1);
            end
        end
    end

    c_raw(cj_idx, :, :, :) = local_c;
end

total_elapsed_seconds = toc(experiment_start_time);
fprintf('\n实验完成，用时 %.2f 分钟。\n', total_elapsed_seconds / 60);

%% 5. 统计
[c_mean, c_std, c_sem] = compute_statistics_4d(c_raw);
c1_mean = squeeze(c_mean(:, :, 1));
c2_mean = squeeze(c_mean(:, :, 2));
delta_c = c2_mean - c1_mean;

c1_sem = squeeze(c_sem(:, :, 1));
c2_sem = squeeze(c_sem(:, :, 2));
delta_sem = sqrt(c1_sem.^2 + c2_sem.^2);

%% 6. 保存
results = struct();
results.description = 'Noise × Threshold scan (c1/c2)';
results.parameters = params;
results.cj_thresholds = cj_thresholds;
results.noise_levels = noise_levels;
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
results.parallel_workers = pool.NumWorkers;
results.matlab_version = version;

save(output_filename, 'results', '-v7.3');
fprintf('结果已保存至: %s\n', output_filename);

if exist(temp_filename, 'file')
    delete(temp_filename);
end

%% 7. 可视化
% 传递 delta_sem 参数以保持接口一致性，虽然在函数中未使用
render_quicklook_figure(cj_thresholds, noise_levels, ...
    c1_mean, c2_mean, delta_c, delta_sem, num_runs, pool.NumWorkers, output_dir);

fprintf('\n实验完成！\n');

%% --------------------- 嵌套函数 ---------------------
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

%% --------------------- 辅助函数 ---------------------
function result = run_single_experiment(params, pulse_count)
    try
        sim = ParticleSimulationWithExternalPulse(params);
        sim.external_pulse_count = pulse_count;
        cascade_size = sim.runSingleExperiment(pulse_count);
        result = cascade_size;
    catch ME
        warning('run_single_experiment:failure', ...
            'Experiment failed (cj=%.3f, sigma=%.3f, pulse=%d): %s', ...
            params.cj_threshold, params.angleNoiseIntensity, pulse_count, ME.message);
        result = NaN;
    end
end

function render_quicklook_figure(cj_thresholds, noise_levels, ...
    c1_mean, c2_mean, delta_c, ~, num_runs, num_workers, output_dir)

    fig = figure('Name', 'Noise × Threshold 扫描结果', ...
        'Position', [100, 100, 1500, 500]);

    subplot(1,3,1);
    imagesc(cj_thresholds, noise_levels, c1_mean');
    axis xy;
    xlabel('c_j threshold');
    ylabel('angle noise σ');
    title('c_1 平均级联规模');
    colorbar;

    subplot(1,3,2);
    imagesc(cj_thresholds, noise_levels, c2_mean');
    axis xy;
    xlabel('c_j threshold');
    ylabel('angle noise σ');
    title('c_2 平均级联规模');
    colorbar;

    subplot(1,3,3);
    imagesc(cj_thresholds, noise_levels, delta_c');
    axis xy;
    xlabel('c_j threshold');
    ylabel('angle noise σ');
    title('\Delta c = c_2 - c_1');
    colorbar;

    sgtitle(sprintf('Δc 扫描 (N=%d, 每组合%d次, %d workers)', ...
        200, num_runs, num_workers));  % TODO: 应该使用 params.N 而不是硬编码的 200

    savefig(fullfile(output_dir, 'heatmap.fig'));
    print(fullfile(output_dir, 'heatmap.png'), '-dpng', '-r300');
    close(fig);
end

function params = default_simulation_parameters()
    params.N = 200;
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
    params.stabilization_steps = 200;
    params.forced_turn_duration = 200;
end

function [mean_values, std_values, sem_values] = compute_statistics_4d(raw_data)
    mean_values = NaN(size(raw_data, 1), size(raw_data, 2), size(raw_data, 4));
    std_values = mean_values;
    sem_values = mean_values;

    for cj_idx = 1:size(raw_data, 1)
        for noise_idx = 1:size(raw_data, 2)
            for pulse_idx = 1:size(raw_data, 4)
                samples = squeeze(raw_data(cj_idx, noise_idx, :, pulse_idx));
                samples = samples(~isnan(samples));
                if isempty(samples)
                    continue;
                end
                mean_values(cj_idx, noise_idx, pulse_idx) = mean(samples);
                std_values(cj_idx, noise_idx, pulse_idx) = std(samples, 0);
                sem_values(cj_idx, noise_idx, pulse_idx) = ...
                    std_values(cj_idx, noise_idx, pulse_idx) / sqrt(numel(samples));
            end
        end
    end
end

function output_dir = ensure_data_directory(timestamp)
    base_dir = fullfile(pwd, 'data', 'experiments', 'noise_threshold_scan');
    if ~isfolder(base_dir)
        mkdir(base_dir);
    end
    output_dir = fullfile(base_dir, timestamp);
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
