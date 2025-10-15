function run_branching_ratio_scan_parallel()
%RUN_BRANCHING_RATIO_SCAN_PARALLEL 并行扫描运动显著性阈值对平均分支比的影响。
%   该脚本与 run_delta_c_parameter_scan_parallel 结构相似，但仅统计级联中的
%   平均分支比 b。对于每个 cj_threshold，运行多次单初发个体 (c1) 实验，并
%   通过事件日志经验计算分支比。

%% 1. 实验设置
fprintf('=================================================\n');
fprintf('   Branching Ratio 参数扫描实验 (并行化版本)\n');
fprintf('=================================================\n\n');

params = default_simulation_parameters();

cj_threshold_min = 0.1;
cj_threshold_max = 8.0;
cj_threshold_step = 0.1;
cj_thresholds = cj_threshold_min:cj_threshold_step:cj_threshold_max;
num_params = numel(cj_thresholds);

num_runs = 100;                                    % 每个阈值重复次数
completed_tasks = 0;                              % 全局已完成任务计数

total_tasks = num_params * num_runs;
fprintf('参数扫描范围: cj_threshold = [%.1f, %.1f], 步长 = %.1f\n', ...
    cj_threshold_min, cj_threshold_max, cj_threshold_step);
fprintf('参数点数量: %d\n', num_params);
fprintf('每个参数重复次数: %d\n', num_runs);
fprintf('总实验次数: %d\n', total_tasks);

%% 2. 并行池配置
pool = configure_parallel_pool();
fprintf('使用并行池: %d workers\n', pool.NumWorkers);

%% 3. 数据结构预分配与计时器
b_raw = NaN(num_params, num_runs);                % 平均分支比

experiment_start_time = tic;
progress_update_timer = tic;
progress_step = max(1, floor(total_tasks / 100));

timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_dir = ensure_data_directory(timestamp);
output_filename = fullfile(output_dir, 'data.mat');
temp_filename = fullfile(output_dir, 'temp.mat');

progress_queue = parallel.pool.DataQueue;
afterEach(progress_queue, @(increment) update_progress(increment));

%% 4. 并行扫描
fprintf('\n开始并行实验...\n');
fprintf('----------------------------------------\n');

shared_params = parallel.pool.Constant(params);

parfor param_idx = 1:num_params
    local_b = NaN(1, num_runs);

    current_params = shared_params.Value;
    current_params.cj_threshold = cj_thresholds(param_idx);

    for run_idx = 1:num_runs
        local_b(run_idx) = run_branching_experiment(current_params, 1);
        send(progress_queue, 1);
    end

    b_raw(param_idx, :) = local_b;
end

fprintf('\n实验执行完毕，用时 %.2f 分钟。\n', toc(experiment_start_time) / 60);
total_elapsed_seconds = toc(experiment_start_time);

%% 5. 统计分析
[b_mean, b_std, b_sem] = compute_statistics(b_raw);
error_count = sum(isnan(b_raw), 2);

%% 6. 保存结果
results = struct();
results.description = 'Branching ratio parameter scan (parallel)';
results.parameters = params;
results.scan_variable = 'cj_threshold';
results.cj_thresholds = cj_thresholds;
results.num_runs = num_runs;
results.b_raw = b_raw;
results.b_mean = b_mean;
results.b_std = b_std;
results.b_sem = b_sem;
results.timestamp = timestamp;
results.date = datetime('now');
results.total_experiments = total_tasks;
results.total_time_seconds = total_elapsed_seconds;
results.total_time_hours = total_elapsed_seconds / 3600;
results.error_count = error_count;
results.parallel_workers = pool.NumWorkers;
results.matlab_version = version;

save(output_filename, 'results', '-v7.3');
fprintf('结果已保存至: %s\n', output_filename);

if exist(temp_filename, 'file')
    delete(temp_filename);
end

%% 7. 快速预览图
fprintf('\n生成快速预览图...\n');
quicklook_prefix = fullfile(output_dir, 'result');
render_quicklook_figure(cj_thresholds, b_mean, b_sem, ...
    error_count, params.N, num_runs, pool.NumWorkers, quicklook_prefix);

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
            interim_results = struct('b_raw', b_raw, ...
                'timestamp', timestamp, 'completed_tasks', completed_tasks);
            save(temp_filename, 'interim_results', '-v7.3');
        end
    end
end

%% 局部函数定义 ------------------------------------------------------------
function result = run_branching_experiment(params, pulse_count)
%RUN_BRANCHING_EXPERIMENT 执行一次实验并返回平均分支比。
    try
        sim = ParticleSimulationWithExternalPulse(params);
        sim.external_pulse_count = pulse_count;
        result = compute_branching_ratio(sim, pulse_count);
    catch ME
        warning('run_branching_experiment:failure', ...
            'Experiment failed (cj=%.3f, pulse=%d): %s', ...
            params.cj_threshold, pulse_count, ME.message);
        result = NaN;
    end
end

function ratio = compute_branching_ratio(sim, pulse_count)
%COMPUTE_BRANCHING_RATIO 运行一次级联并统计平均分支比。

    sim.resetCascadeTracking();
    sim.initializeParticles();

    original_pulse_count = sim.external_pulse_count;
    sim.external_pulse_count = pulse_count;
    sim.current_step = 0;

    parent_candidates = false(sim.N, 1);
    children_count = zeros(sim.N, 1);

    % 稳定期：不统计
    for step = 1:sim.stabilization_steps
        sim.step();
    end

    max_cascade_steps = 200;
    for step = 1:max_cascade_steps
        prev_active = sim.isActive;
        sim.step();

        newly_active = sim.isActive & ~prev_active;
        if any(newly_active)
            activated_indices = find(newly_active);
            for idx = activated_indices'
                parent_candidates(idx) = true;
                parent = sim.src_ids{idx};
                if ~isempty(parent)
                    parent_idx = parent(1);
                    if parent_idx >= 1 && parent_idx <= sim.N
                        parent_idx = round(parent_idx);
                        children_count(parent_idx) = children_count(parent_idx) + 1;
                        parent_candidates(parent_idx) = true;
                    end
                end
            end
        end

        if sim.isCascadeComplete()
            break;
        end
    end

    parents = find(parent_candidates);
    if isempty(parents)
        ratio = 0;
    else
        ratio = sum(children_count(parents)) / numel(parents);
    end

    sim.external_pulse_count = original_pulse_count;
end

function params_out = default_simulation_parameters()
%DEFAULT_SIMULATION_PARAMETERS 生成与 Delta_c 扫描一致的默认参数。
    params_out.N = 200;
    params_out.rho = 1;
    params_out.v0 = 1;
    params_out.angleUpdateParameter = 10;
    params_out.angleNoiseIntensity = 0;
    params_out.T_max = 400;
    params_out.dt = 0.1;
    params_out.radius = 5;
    params_out.deac_threshold = 0.1745;
    params_out.cj_threshold = 1;
    params_out.fieldSize = 50;
    params_out.initDirection = pi / 4;
    params_out.useFixedField = true;
    params_out.stabilization_steps = 200;
    params_out.forced_turn_duration = 200;
end

function [mean_values, std_values, sem_values] = compute_statistics(raw_data)
%COMPUTE_STATISTICS 计算均值、标准差与标准误差。
    mean_values = NaN(size(raw_data, 1), 1);
    std_values = mean_values;
    sem_values = mean_values;

    for idx = 1:size(raw_data, 1)
        valid_samples = raw_data(idx, ~isnan(raw_data(idx, :)));
        if isempty(valid_samples)
            continue;
        end
        mean_values(idx) = mean(valid_samples);
        std_values(idx) = std(valid_samples, 0);
        sem_values(idx) = std_values(idx) / sqrt(numel(valid_samples));
    end
end

function output_dir = ensure_data_directory(timestamp)
%ENSURE_DATA_DIRECTORY 创建实验数据目录（按时间戳组织）。
    output_dir = fullfile(pwd, 'data', 'experiments', 'branching_ratio_scan', timestamp);
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
end

function pool = configure_parallel_pool()
%CONFIGURE_PARALLEL_POOL 确保本地并行池已就绪，如无则按上限启动。
    if ~license('test', 'Distrib_Computing_Toolbox')
        error('Parallel Computing Toolbox license is required.');
    end

    pool = gcp('nocreate');
    if isempty(pool)
        cluster = parcluster('local');
        max_workers = min(cluster.NumWorkers, 40);
        if max_workers < 1
            error('No available workers for parallel pool.');
        end
        pool = parpool(cluster, max_workers);
    end
end

function render_quicklook_figure(thresholds, b_mean, b_sem, ...
    error_count, N, num_runs, num_workers, output_prefix)
%RENDER_QUICKLOOK_FIGURE 生成分支比扫描的双联图预览。

    fig = figure('Name', 'Branching Ratio 并行扫描结果', ...
        'Position', [100, 100, 900, 400]);

    % (1) 平均分支比趋势
    subplot(1, 2, 1);
    errorbar(thresholds, b_mean, b_sem, 'b.-', 'LineWidth', 1.5);
    xlabel('c_j threshold');
    ylabel('平均分支比 b');
    title('平均分支比 vs 运动显著性阈值');
    grid on;

    % (2) 失败次数统计
    subplot(1, 2, 2);
    bar(thresholds, error_count, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none');
    xlabel('c_j threshold');
    ylabel('失败次数');
    title('每个阈值的实验失败数');
    grid on;

    sgtitle(sprintf('分支比分析 (N=%d, %d次重复, %d workers)', ...
        N, num_runs, num_workers));

    savefig([output_prefix, '.fig']);
    print([output_prefix, '.png'], '-dpng', '-r300');
    close(fig);
end
