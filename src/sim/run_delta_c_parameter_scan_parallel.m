function run_delta_c_parameter_scan_parallel()
%RUN_DELTA_C_PARAMETER_SCAN_PARALLEL 并行扫描运动显著性阈值对 Δc 的影响。
%   主控函数执行以下步骤：准备仿真参数、配置并行池、预分配结果矩阵、
%   使用 parfor 调度成对的 c1/c2 级联实验、汇总统计量、保存结果并绘制
%   快速预览图。脚本假定 MATLAB 并行计算工具箱可用。

%% 1. 实验设置与参数范围
fprintf('=================================================\n');
fprintf('     Delta_c 参数扫描实验 (并行化版本)\n');
fprintf('=================================================\n\n');

params = default_simulation_parameters();

cj_threshold_min = 0.1;
cj_threshold_max = 8.0;
cj_threshold_step = 0.1;
cj_thresholds = cj_threshold_min:cj_threshold_step:cj_threshold_max;
num_params = numel(cj_thresholds);

num_runs = 50;                                    % 每个阈值重复次数
trials_per_param = num_runs * 2;                  % c1 与 c2 各一次
completed_tasks = 0;                              % 全局已完成任务计数

total_tasks = num_params * trials_per_param;
fprintf('参数扫描范围: cj_threshold = [%.1f, %.1f], 步长 = %.1f\n', ...
    cj_threshold_min, cj_threshold_max, cj_threshold_step);
fprintf('参数点数量: %d\n', num_params);
fprintf('每个参数重复次数: %d (c1/c2 各一次)\n', num_runs);
fprintf('总实验次数: %d\n', total_tasks);

%% 2. 并行池配置（强制并行模式）
pool = configure_parallel_pool();
fprintf('使用并行池: %d workers\n', pool.NumWorkers);

%% 3. 数据结构预分配与计时器初始化
c1_raw = NaN(num_params, num_runs);               % 单外源实验结果
c2_raw = NaN(num_params, num_runs);               % 双外源实验结果

experiment_start_time = tic;                      % 总耗时计时器
progress_update_timer = tic;                      % 进度刷新计时器
progress_step = max(1, floor(total_tasks / 100));  % 至少 1% 更新一次

% 生成输出路径及临时文件名，以便崩溃时恢复
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_dir = ensure_data_directory();
output_filename = fullfile(output_dir, ...
    sprintf('delta_c_scan_parallel_%s.mat', timestamp));
temp_filename = fullfile(output_dir, ...
    sprintf('delta_c_scan_parallel_%s_temp.mat', timestamp));

% DataQueue 用于 parfor worker 通知主线程进度
progress_queue = parallel.pool.DataQueue;
afterEach(progress_queue, @(increment) update_progress(increment));

%% 4. 使用 parfor 执行参数扫描
fprintf('\n开始并行实验...\n');
fprintf('----------------------------------------\n');

shared_params = parallel.pool.Constant(params);  % 广播基础参数

parfor param_idx = 1:num_params
    local_c1 = NaN(1, num_runs);
    local_c2 = NaN(1, num_runs);

    current_params = shared_params.Value;
    current_params.cj_threshold = cj_thresholds(param_idx);

    for run_idx = 1:num_runs
        local_c1(run_idx) = run_single_experiment(current_params, 1);
        send(progress_queue, 1);

        local_c2(run_idx) = run_single_experiment(current_params, 2);
        send(progress_queue, 1);
    end

    c1_raw(param_idx, :) = local_c1;
    c2_raw(param_idx, :) = local_c2;
end

fprintf('\n实验执行完毕，用时 %.2f 分钟。\n', toc(experiment_start_time) / 60);

total_elapsed_seconds = toc(experiment_start_time);

%% 5. 统计分析
[c1_mean, c1_std, c1_sem] = compute_statistics(c1_raw);
[c2_mean, c2_std, c2_sem] = compute_statistics(c2_raw);
% Δc 衡量双外源相较单外源所带来的级联增益
delta_c = c2_mean - c1_mean;

% 统计各阈值下的失败次数，为成功率与可靠性分析提供依据
error_count = [sum(isnan(c1_raw), 2), sum(isnan(c2_raw), 2)];

%% 6. 结果保存
results = struct();
results.description = 'Delta_c parameter scan (parallel only)';
results.parameters = params;
results.scan_variable = 'cj_threshold';
results.cj_thresholds = cj_thresholds;
results.num_runs = num_runs;
results.c1_raw = c1_raw;
results.c2_raw = c2_raw;
results.c1_mean = c1_mean;
results.c2_mean = c2_mean;
results.c1_std = c1_std;
results.c2_std = c2_std;
results.c1_sem = c1_sem;
results.c2_sem = c2_sem;
results.delta_c = delta_c;
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

%% 7. 摘要输出与快速预览图
fprintf('\n=================================================\n');
fprintf('                实验完成摘要\n');
fprintf('=================================================\n');
fprintf('总耗时: %.2f 小时\n', total_elapsed_seconds / 3600);
fprintf('并行工作线程: %d\n', pool.NumWorkers);

success_rate_c1 = 1 - sum(error_count(:, 1)) / (num_runs * num_params);
success_rate_c2 = 1 - sum(error_count(:, 2)) / (num_runs * num_params);
fprintf('成功率: %.1f%% (c1), %.1f%% (c2)\n', ...
    100 * success_rate_c1, 100 * success_rate_c2);

[max_delta_c, max_idx] = max(delta_c);
optimal_cj = cj_thresholds(max_idx);
fprintf('\nDelta_c 峰值: %.4f (在 cj_threshold = %.2f)\n', max_delta_c, optimal_cj);

quicklook_figure_path = fullfile(output_dir, ...
    sprintf('delta_c_scan_parallel_%s', timestamp));
render_quicklook_figure(cj_thresholds, c1_mean, c1_sem, ...
    c2_mean, c2_sem, delta_c, optimal_cj, max_delta_c, error_count, ...
    params.N, num_runs, pool.NumWorkers, quicklook_figure_path);

fprintf('\n实验完成！\n');

%% 嵌套辅助函数 ------------------------------------------------------------
    function params_out = default_simulation_parameters()
        % 仿真规模与动力学参数
        params_out.N = 200;
        params_out.rho = 1;
        params_out.v0 = 1;
        params_out.angleUpdateParameter = 10;
        params_out.angleNoiseIntensity = 0;
        params_out.T_max = 400;
        params_out.dt = 0.1;
        params_out.radius = 5;
        % 行为相关阈值（cj_threshold 将在扫描循环内覆盖）
        params_out.deac_threshold = 0.1745;
        params_out.cj_threshold = 1;
        % 场地与初始条件
        params_out.fieldSize = 50;
        params_out.initDirection = pi / 4;
        params_out.useFixedField = true;
        % 外源激活策略参数
        params_out.stabilization_steps = 200;
        params_out.forced_turn_duration = 200;
    end

    function [mean_values, std_values, sem_values] = compute_statistics(raw_data)
        % 针对每一行参数点，剔除 NaN 后计算均值、标准差与标准误差
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

    function update_progress(increment)
        % DataQueue 回调：累加已完成任务数并输出进度与 ETA
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
            interim_results = struct('c1_raw', c1_raw, 'c2_raw', c2_raw, ...
                'timestamp', timestamp, 'completed_tasks', completed_tasks);
            save(temp_filename, 'interim_results', '-v7.3');
        end
    end
end

%% 局部函数定义 ------------------------------------------------------------
function result = run_single_experiment(params, exp_type)
%RUN_SINGLE_EXPERIMENT 执行一次 c1/c2 级联试验并返回归一化级联规模。
    try
        sim = ParticleSimulationWithExternalPulse(params);
        sim.external_pulse_count = exp_type;
        cascade_size = sim.runSingleExperiment(exp_type);
        result = (cascade_size * params.N - exp_type) / params.N;
    catch ME
        warning('run_single_experiment:failure', ...
            'Experiment failed (cj=%.3f, exp_type=%d): %s', ...
            params.cj_threshold, exp_type, ME.message);
        result = NaN;
    end
end

function output_dir = ensure_data_directory()
%ENSURE_DATA_DIRECTORY 若 data 目录不存在则创建。
    output_dir = fullfile(pwd, 'data');
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

function render_quicklook_figure(thresholds, c1_mean, c1_sem, ...
    c2_mean, c2_sem, delta_c, optimal_cj, max_delta_c, error_count, ...
    N, num_runs, num_workers, output_prefix)
%RENDER_QUICKLOOK_FIGURE 生成扫描结果的三联图并保存。

    figure_handle = figure('Name', 'Delta_c 并行扫描结果', ...
        'Position', [100, 100, 1200, 400]);

    % (1) 平均级联规模趋势
    subplot(1, 3, 1);
    hold on;
    errorbar(thresholds, c1_mean, c1_sem, 'b.-', 'LineWidth', 1.5, ...
        'DisplayName', 'c_1');
    errorbar(thresholds, c2_mean, c2_sem, 'r.-', 'LineWidth', 1.5, ...
        'DisplayName', 'c_2');
    xlabel('c_j threshold');
    ylabel('平均级联规模');
    title('级联规模 vs 运动显著性阈值');
    legend('Location', 'best');
    grid on;
    hold off;

    % (2) Δc 曲线及峰值标记
    subplot(1, 3, 2);
    plot(thresholds, delta_c, 'k.-', 'LineWidth', 2);
    hold on;
    plot(optimal_cj, max_delta_c, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    xlabel('c_j threshold');
    ylabel('\Delta c = c_2 - c_1');
    title('集群敏感性 vs 运动显著性阈值');
    grid on;
    hold off;

    % (3) 失败次数统计
    subplot(1, 3, 3);
    bar(1:2, [sum(error_count(:, 1)), sum(error_count(:, 2))]);
    set(gca, 'XTickLabel', {'c1实验', 'c2实验'});
    ylabel('失败次数');
    title('实验失败统计');

    sgtitle(sprintf('Delta_c 并行扫描结果 (N=%d, %d次重复, %d workers)', ...
        N, num_runs, num_workers));

    savefig([output_prefix, '.fig']);
    print([output_prefix, '.png'], '-dpng', '-r300');
    close(figure_handle);
end
