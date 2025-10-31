function run_delta_c_m_vs_1_scan_parallel()
%RUN_DELTA_C_M_VS_1_SCAN_PARALLEL 并行扫描运动显著性阈值对Δc_{m,1}的影响
% 扫描cj_threshold、初发个体数量并生成统计结果与快览图

%% 1. 实验设定
fprintf('=================================================\n');
fprintf('   Delta_c (1 vs m) 参数扫描实验 (并行化版本)\n');
fprintf('=================================================\n\n');

addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..')));
params = default_simulation_parameters();

% 运动显著性阈值扫描参数
cj_threshold_min = 0;
cj_threshold_max = 5.0;
cj_threshold_step = 0.1;
cj_thresholds = cj_threshold_min:cj_threshold_step:cj_threshold_max;
num_params = numel(cj_thresholds);

% 实验重复和初发个体配置
num_runs = 50;                    % 每个参数点重复50次
m_values = 2:10;                 % 与c1对比的初发个体数量
pulse_counts = [1, m_values];     % [1,2,3,...,10]
num_pulses = numel(pulse_counts);

% 随机数种子管理
BASE_SEED = 202501;               % 基础随机数种子
rng(BASE_SEED, 'threefry');

% 实验规模计算
trials_per_param = num_runs * num_pulses;
completed_tasks = 0;
total_tasks = num_params * trials_per_param;

% 随机数种子矩阵：确保每个实验都有独立的随机数序列
seed_matrix = BASE_SEED + reshape(0:(total_tasks - 1), num_params, num_runs, num_pulses);

fprintf('参数扫描范围: [%.1f, %.1f], 步长 %.1f，共 %d 个参数点\n', ...
    cj_threshold_min, cj_threshold_max, cj_threshold_step, num_params);
fprintf('初发个体数: %s (共 %d 种)\n', mat2str(pulse_counts), num_pulses);
fprintf('每个参数重复次数: %d，总实验次数: %d\n', num_runs, total_tasks);

%% 2. 并行池配置
pool = configure_parallel_pool();
fprintf('使用并行池: %d workers\n', pool.NumWorkers);

%% 3. 预分配
c_raw = NaN(num_params, num_runs, num_pulses);  % 级联规模原始数据
b_raw = NaN(num_params, num_runs, num_pulses);  % 分支比原始数据

experiment_start_time = tic;
progress_update_timer = tic;
progress_step = max(1, floor(total_tasks / 100));

timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_dir = ensure_data_directory(timestamp, params.N);
output_filename = fullfile(output_dir, 'data.mat');
temp_filename = fullfile(output_dir, 'temp.mat');

progress_queue = parallel.pool.DataQueue;
afterEach(progress_queue, @(inc) update_progress(inc));

%% 4. 核心并行仿真实验

fprintf('\n开始并行实验...\n');
fprintf('----------------------------------------\n');

% 通过Constant共享只读参数，降低worker初始化成本
shared_params = parallel.pool.Constant(params);
shared_pulse_counts = parallel.pool.Constant(pulse_counts);

parfor param_idx = 1:num_params
    local_c = NaN(num_runs, num_pulses);
    local_b = NaN(num_runs, num_pulses);

    current_params = shared_params.Value;
    current_params.cj_threshold = cj_thresholds(param_idx);
    local_pulse_counts = shared_pulse_counts.Value;

    for run_idx = 1:num_runs
        for pulse_idx = 1:num_pulses
            pulse = local_pulse_counts(pulse_idx);
            seed_value = seed_matrix(param_idx, run_idx, pulse_idx);

            [cascade_size, branching_ratio] = run_single_experiment(current_params, pulse, seed_value);

            local_c(run_idx, pulse_idx) = cascade_size;
            local_b(run_idx, pulse_idx) = branching_ratio;

            send(progress_queue, 1);
        end
    end

    c_raw(param_idx, :, :) = local_c;
    b_raw(param_idx, :, :) = local_b;
end

% 计算总实验耗时并输出
total_elapsed_seconds = toc(experiment_start_time);
fprintf('\n实验执行完毕，用时 %.2f 分钟。\n', total_elapsed_seconds / 60);

%% 5. 统计分析和Delta_c计算
% 统计三维结果并计算Delta_c
[c_mean, c_std, c_sem] = compute_statistics_3d(c_raw);  % 级联规模统计
[b_mean, b_std, b_sem] = compute_statistics_3d(b_raw);  % 分支比统计

% NaN表示失败的实验
error_count = sum(isnan(c_raw), 2);
error_count = reshape(error_count, num_params, num_pulses);

% 计算成功率与基准c_1
success_rate_matrix = 1 - error_count / num_runs;
success_rate_per_pulse = 1 - sum(error_count, 1) / (num_runs * num_params);

c1_mean = c_mean(:, 1);   % 基准级联规模均值 [参数点, 1]
c1_sem = c_sem(:, 1);     % 基准级联规模标准误差 [参数点, 1]

% 计算Delta_c及其误差
delta_c = c_mean(:, 2:end) - c1_mean;
delta_sem = sqrt(c_sem(:, 2:end).^2 + c1_sem.^2);

%% 6. 结果数据结构化和保存
results = struct();

results.description = 'Delta_c parameter scan (1 vs m)';
results.parameters = params;
results.scan_variable = 'cj_threshold';

results.cj_thresholds = cj_thresholds;
results.num_runs = num_runs;
results.pulse_counts = pulse_counts;

results.c_raw = c_raw;      % 原始级联数据 [参数, 重复, 脉冲]
results.c_mean = c_mean;    % 级联规模均值 [参数, 脉冲]
results.c_std = c_std;      % 级联规模标准差 [参数, 脉冲]
results.c_sem = c_sem;      % 级联规模标准误差 [参数, 脉冲]

results.branching_raw = b_raw;      % 原始分支比数据
results.branching_mean = b_mean;    % 分支比均值
results.branching_std = b_std;      % 分支比标准差
results.branching_sem = b_sem;      % 分支比标准误差

results.delta_c = delta_c;          % 多初发敏感性指标
results.delta_sem = delta_sem;      % Delta_c标准误差

results.timestamp = timestamp;
results.date = datetime('now');

results.total_experiments = total_tasks;
results.total_time_seconds = total_elapsed_seconds;    % 总耗时（秒）
results.total_time_hours = total_elapsed_seconds / 3600; % 总耗时（小时）

results.error_count = error_count;

results.success_rate_matrix = success_rate_matrix;
results.success_rate_per_pulse = success_rate_per_pulse;
results.parallel_workers = pool.NumWorkers;
results.matlab_version = version;
results.base_seed = BASE_SEED;

save(output_filename, 'results', '-v7.3');
fprintf('结果已保存至: %s\n', output_filename);

if exist(temp_filename, 'file')
    delete(temp_filename);
end

%% 7. 实验摘要输出和结果分析
fprintf('\n=================================================\n');
fprintf('                实验完成摘要\n');
fprintf('=================================================\n');

fprintf('总耗时: %.2f 小时\n', total_elapsed_seconds / 3600);

fprintf('成功率范围: %.1f%% ~ %.1f%% (各脉冲)\n', ...
    100 * min(success_rate_per_pulse), 100 * max(success_rate_per_pulse));

[max_delta, linear_idx] = max(delta_c(:));

[max_param_idx, max_m_idx] = ind2sub(size(delta_c), linear_idx);

optimal_cj = cj_thresholds(max_param_idx);
optimal_m = pulse_counts(max_m_idx + 1);  % +1因为delta_c排除了第一列(c1)

fprintf('Δc 峰值: %.4f (cj_threshold = %.2f, m = %d)\n', ...
    max_delta, optimal_cj, optimal_m);

fprintf('平均分支比范围: [%.3f, %.3f]\n', min(b_mean(:, 1)), max(b_mean(:, 1)));

quicklook_path = fullfile(output_dir, 'result');
render_quicklook_figure(cj_thresholds, pulse_counts, c_mean, c_sem, ...
    delta_c, delta_sem, error_count, success_rate_matrix, num_runs, pool.NumWorkers, quicklook_path);

fprintf('\n实验完成！\n');

%% 嵌套函数 ---------------------------------------------------------------
    function update_progress(increment)
        %UPDATE_PROGRESS 更新实验进度并按需保存中间结果
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
function [cascade_size, branching_ratio] = run_single_experiment(params, pulse_count, seed_value)
%RUN_SINGLE_EXPERIMENT 执行单次级联实验，输出级联规模与分支比

try
    cleanup_stream = [];

    if nargin >= 3 && ~isempty(seed_value)
        parent_stream = RandStream.getGlobalStream;
        child_stream = RandStream('Threefry', 'Seed', seed_value);
        RandStream.setGlobalStream(child_stream);
        cleanup_stream = onCleanup(@() RandStream.setGlobalStream(parent_stream));
    end

    sim = ParticleSimulationWithExternalPulse(params);
    sim.external_pulse_count = pulse_count;

    [branching_ratio, cascade_size] = compute_branching_ratio_with_cascade(sim, pulse_count);

    if ~isempty(cleanup_stream)
        clear cleanup_stream;
    end

catch ME
    warning('run_single_experiment:failure', ...
        'Experiment failed (cj=%.3f, pulse=%d): %s', ...
        params.cj_threshold, pulse_count, ME.message);

    cascade_size = NaN;
    branching_ratio = NaN;
end
end

function [ratio, cascade_size] = compute_branching_ratio_with_cascade(sim, pulse_count)
%COMPUTE_BRANCHING_RATIO_WITH_CASCADE 返回分支比与级联规模

sim.resetCascadeTracking();

sim.initializeParticles();

original_pulse_count = sim.external_pulse_count;

sim.external_pulse_count = pulse_count;

sim.current_step = 0;

parent_flags = false(sim.N, 1);

children_count = zeros(sim.N, 1);

counting_enabled = false;

tracking_deadline = sim.T_max;

track_steps_after_trigger = 100; % 触发后追踪步数

for step = 1:sim.stabilization_steps
    sim.step();
end

max_remaining_steps = max(1, sim.T_max - sim.current_step);

for step = 1:max_remaining_steps
    prev_active = sim.isActive;

    sim.step();

    if ~counting_enabled && sim.external_pulse_triggered
        counting_enabled = true;
        tracking_deadline = min(sim.T_max, sim.current_step + track_steps_after_trigger);
    end

    newly_active = sim.isActive & ~prev_active;

    if counting_enabled && sim.current_step <= tracking_deadline && any(newly_active)
        activated_indices = find(newly_active);

        for idx = activated_indices'
            parent_flags(idx) = true;

            parent = sim.src_ids{idx};

            if ~isempty(parent)
                parent_idx = parent(1);

                if parent_idx >= 1 && parent_idx <= sim.N
                    parent_idx = round(parent_idx);
                    children_count(parent_idx) = children_count(parent_idx) + 1;
                    parent_flags(parent_idx) = true;
                end
            end
        end
    end

    if counting_enabled && (~sim.cascade_active || sim.current_step >= tracking_deadline)
        break;
    end
end

if ~counting_enabled
    ratio = 0;
else
    parents = find(parent_flags);

    if isempty(parents)
        ratio = 0;
    else
        ratio = sum(children_count(parents)) / numel(parents);
    end
end

cascade_size = sum(sim.everActivated) / sim.N;

sim.external_pulse_count = original_pulse_count;
end

function render_quicklook_figure(thresholds, pulse_counts, c_mean, c_sem, ...
    delta_c, delta_sem, error_count, success_rate_matrix, num_runs, num_workers, output_prefix)
%RENDER_QUICKLOOK_FIGURE 绘制Delta_c (1 vs m) 三联图并保存FIG/PNG

fig = figure('Name', 'Delta_c (1 vs m) 扫描结果', ...
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
    200, num_runs, num_workers));

savefig([output_prefix, '.fig']);
print([output_prefix, '.png'], '-dpng', '-r300');
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

function output_dir = ensure_data_directory(timestamp, N)
base_dir = fullfile(pwd, 'data', 'experiments', 'delta_c_m_vs_1_scan');
if ~isfolder(base_dir)
    mkdir(base_dir);
end
if nargin < 2 || isempty(N)
    folder_name = timestamp;
else
    folder_name = sprintf('N%d_%s', N, timestamp);
end
output_dir = fullfile(base_dir, folder_name);
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
