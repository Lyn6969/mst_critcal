function run_generation_single_pulse_MT_scan_debug(max_partial_generations)
%RUN_GENERATION_SINGLE_PULSE_MT_SCAN_DEBUG
% 在“单初发个体、零噪声”的条件下，扫描激活阈值 M_T (cj_threshold)，
% 使用“按世代分层 + 首次激活”的分支比定义，计算并绘制 b(M_T) 曲线，
% 同时将所有用于绘图的数据和关键中间量保存为 JSON，便于离线仔细检查。
%
% 主要功能：
%   1. 固定 N 和其他参数，仅改变 cj_threshold。
%   2. 每个参数点重复 num_runs 次实验。
%   3. 使用“世代 + 首次激活”算法计算单次实验的分支比 b_run 与世代规模 Z_g。
%   4. 对每个 M_T 计算 b 的均值 / 标准差 / 标准误，并绘制 b(M_T) 曲线。
%   5. 在对应的数据目录下保存：
%        - MATLAB 格式结果（.mat）
%        - 完整 JSON 文件（包括所有 b_run 和每次实验的 Z_g）
%
% 使用方式：
%   在 MATLAB 命令行中直接运行（默认只看前两代）：
%       run_generation_single_pulse_MT_scan_debug
%   或者显式指定只看前 K 代（K>=1）：
%       run_generation_single_pulse_MT_scan_debug(K)
%
% 注意：
%   - 仅考虑单初发个体（pulse_count = 1）。
%   - 噪声强度 angleNoiseIntensity 固定为 0。

    %% 0. 初始化与参数设置
    clear; clc; close all;

    % 将项目根目录加入路径
    script_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(fileparts(script_dir)); % 向上两级
    addpath(genpath(project_root));

    % 解析仅前若干代的分支比参数（默认 K = 2）
    if nargin < 1 || isempty(max_partial_generations)
        max_partial_generations = 2;
    end
    max_partial_generations = max(1, round(max_partial_generations));

    % 基本仿真参数
    params = default_simulation_parameters();
    params.N = 200;                    % 可按需修改
    params.angleNoiseIntensity = 0;    % 噪声强度固定为 0

    % 扫描的 M_T (cj_threshold) 范围
    cj_thresholds = 0:0.1:5;
    num_params = numel(cj_thresholds);

    % 单初发个体
    pulse_count = 1;

    % 每个参数点重复次数
    num_runs = 50;

    fprintf('=== 单初发个体世代分层分支比调试扫描 ===\n');
    fprintf('N = %d, 噪声 = 0, 初发个体 = %d\n', params.N, pulse_count);
    fprintf('M_T 扫描范围: [%.1f, %.1f], 步长 %.1f，共 %d 个参数点，每点 %d 次实验\n', ...
        cj_thresholds(1), cj_thresholds(end), cj_thresholds(2) - cj_thresholds(1), ...
        num_params, num_runs);
    fprintf('仅前 %d 代分支比将单独统计并绘制。\n', max_partial_generations);

    %% 1. 配置并行池
    pool = configure_parallel_pool();
    fprintf('使用并行池: %d workers\n', pool.NumWorkers);

    %% 2. 结果预分配
    % 全局分支比分布：每个参数点 × 每次实验
    b_raw = NaN(num_params, num_runs);
    % 仅前若干代分支比（0→1→K）：每个参数点 × 每次实验
    b_partial_raw = NaN(num_params, num_runs);
    % 级联规模分布
    c_raw = NaN(num_params, num_runs);
    % 世代规模 Z_g：维度不固定，使用 cell 保存
    generation_counts = cell(num_params, num_runs);  % 每格是一个 [G+1,1] 向量

    %% 3. 输出目录与文件名
    timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    base_dir = fullfile(project_root, 'data', 'experiments', 'generation_single_pulse_scan');
    if ~isfolder(base_dir)
        mkdir(base_dir);
    end
    folder_name = sprintf('N%d_single_pulse_eta_0p000_%s', params.N, timestamp);
    output_dir = fullfile(base_dir, folder_name);
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end

    mat_path = fullfile(output_dir, 'data.mat');
    json_path = fullfile(output_dir, 'generation_single_pulse_results.json');

    %% 4. 主循环：逐个 M_T、逐次实验（并行）
    shared_params = parallel.pool.Constant(params);

    experiment_start_time = tic;
    fprintf('开始并行实验...\n');

    parfor p_idx = 1:num_params
        local_params = shared_params.Value;
        local_params.cj_threshold = cj_thresholds(p_idx);

        local_b = NaN(1, num_runs);
        local_b_partial = NaN(1, num_runs);
        local_c = NaN(1, num_runs);
        local_generation = cell(1, num_runs);

        fprintf('--- M_T = %.2f (%d/%d) ---\n', local_params.cj_threshold, p_idx, num_params);

        for run_idx = 1:num_runs
            try
                sim = ParticleSimulationWithExternalPulse(local_params);
                sim.external_pulse_count = pulse_count;

                [b_run, b_partial_run, c_run, Z_vec] = compute_generation_branching_ratio_single(sim, max_partial_generations);

                local_b(run_idx) = b_run;
                local_b_partial(run_idx) = b_partial_run;
                local_c(run_idx) = c_run;
                local_generation{run_idx} = Z_vec;
            catch ME
                warning('单次实验失败: M_T=%.3f, run=%d, 错误：%s', ...
                    local_params.cj_threshold, run_idx, ME.message);
                local_b(run_idx) = NaN;
                local_c(run_idx) = NaN;
                local_generation{run_idx} = [];
            end
        end

        b_raw(p_idx, :) = local_b;
        b_partial_raw(p_idx, :) = local_b_partial;
        c_raw(p_idx, :) = local_c;
        generation_counts(p_idx, :) = local_generation;
    end

    total_elapsed_seconds = toc(experiment_start_time);
    fprintf('全部实验完成，总耗时 %.2f 分钟。\n', total_elapsed_seconds / 60);

    %% 5. 统计分支比（按 M_T）
    [b_mean, b_std, b_sem] = compute_statistics_2d(b_raw);
    [b_partial_mean, b_partial_std, b_partial_sem] = compute_statistics_2d(b_partial_raw);

    %% 6. 保存 MATLAB 格式结果
    results = struct();
    results.description = sprintf(['Single-pulse generation-based branching ratio scan ', ...
        '(N=%d, eta=0)'], params.N);
    results.parameters = params;
    results.cj_thresholds = cj_thresholds;
    results.num_runs = num_runs;
    results.pulse_count = pulse_count;
    results.b_raw = b_raw;
    results.b_mean = b_mean;
    results.b_std = b_std;
    results.b_sem = b_sem;
    % 仅前若干代分支比（0→1→K）的统计
    results.partial_generations = max_partial_generations;
    results.b_partial_raw = b_partial_raw;
    results.b_partial_mean = b_partial_mean;
    results.b_partial_std = b_partial_std;
    results.b_partial_sem = b_partial_sem;
    results.c_raw = c_raw;
    results.generation_counts = generation_counts;
    results.timestamp = timestamp;
    results.total_time_seconds = total_elapsed_seconds;
    results.total_time_hours = total_elapsed_seconds / 3600;

    save(mat_path, 'results', '-v7.3');
    fprintf('MAT 文件已保存：%s\n', mat_path);

    %% 7. 导出 JSON（包含所有中间数据）
    json_struct = struct();
    json_struct.description = results.description;
    json_struct.N = params.N;
    json_struct.pulse_count = pulse_count;
    json_struct.num_runs = num_runs;
    json_struct.cj_thresholds = cj_thresholds;
    json_struct.b_raw = b_raw;
    json_struct.b_mean = b_mean;
    json_struct.b_std = b_std;
    json_struct.b_sem = b_sem;
    json_struct.partial_generations = max_partial_generations;
    json_struct.b_partial_raw = b_partial_raw;
    json_struct.b_partial_mean = b_partial_mean;
    json_struct.b_partial_std = b_partial_std;
    json_struct.b_partial_sem = b_partial_sem;
    json_struct.c_raw = c_raw;

    % 将每个 (M_T, run) 的 Z_g 也保存进 JSON，方便深入检查：
    % generation_data 是一个长度为 num_params 的结构数组，
    % 每个元素包含当前 M_T 下各次实验的 Z_g 列表。
    generation_data = cell(num_params, 1);
    for p_idx = 1:num_params
        entry = struct();
        entry.cj_threshold = cj_thresholds(p_idx);
        run_cells = cell(num_runs, 1);
        for run_idx = 1:num_runs
            Z_vec = generation_counts{p_idx, run_idx};
            if isempty(Z_vec)
                run_cells{run_idx} = [];
            else
                run_cells{run_idx} = Z_vec(:).'; % 行向量形式
            end
        end
        entry.Z_per_run = run_cells;
        generation_data{p_idx} = entry;
    end
    json_struct.generation_data = generation_data;

    json_text = jsonencode(json_struct, 'PrettyPrint', true);
    fid = fopen(json_path, 'w');
    if fid == -1
        warning('无法创建 JSON 文件：%s', json_path);
    else
        fwrite(fid, json_text, 'char');
        fclose(fid);
        fprintf('JSON 文件已保存：%s\n', json_path);
    end

    %% 8. 绘制 b(M_T) 曲线（简单调试图）
    fig = figure('Position', [150, 150, 500, 300], 'Color', 'white');
    ax = axes('Parent', fig);
    hold(ax, 'on');

    errorbar(ax, cj_thresholds, b_mean, b_sem, 'o-', ...
        'LineWidth', 1.5, 'MarkerSize', 4, 'Color', [0.1, 0.4, 0.8], ...
        'DisplayName', '全局 b');
    label_partial = sprintf('前 %d 代 b_{0-%d}', max_partial_generations, max_partial_generations);
    errorbar(ax, cj_thresholds, b_partial_mean, b_partial_sem, 's-', ...
        'LineWidth', 1.5, 'MarkerSize', 4, 'Color', [0.8, 0.2, 0.2], ...
        'DisplayName', label_partial);
    yline(ax, 1.0, '--', 'Color', [0.8, 0.6, 0.1], 'LineWidth', 1.5);

    xlabel(ax, 'M_T (cj threshold)', 'FontSize', 12);
    ylabel(ax, '世代分层分支比 b', 'FontSize', 12);
    title(ax, sprintf('N = %d, 初发个体 = %d, 噪声 = 0', params.N, pulse_count), 'FontSize', 12);
    legend(ax, 'Location', 'best');
    grid(ax, 'on');
    box(ax, 'on');

    hold(ax, 'off');

    png_path = fullfile(output_dir, 'generation_single_pulse_branching_curve.png');
    saveas(fig, png_path);
    fprintf('b(M_T) 曲线已保存为：%s\n', png_path);
end

%% 辅助函数：单次实验中按世代分层计算分支比
function [ratio, ratio_partial, cascade_size, Z_vec] = compute_generation_branching_ratio_single(sim, max_partial_generations)
%COMPUTE_GENERATION_BRANCHING_RATIO_SINGLE
% 对给定的 ParticleSimulationWithExternalPulse 对象，运行一次完整实验，
% 并使用“按世代分层 + 首次激活”的方式估计分支比，同时返回世代规模 Z_g。

    sim.resetCascadeTracking();
    sim.initializeParticles();

    original_pulse_count = sim.external_pulse_count;
    sim.current_step = 0;

    N = sim.N;

    % 每个节点是否已经作为“事件”被计入过（首次激活）
    first_activated = false(N, 1);
    % 每个节点的世代编号（首次激活时确定），NaN 表示尚未进入家谱树
    generation = NaN(N, 1);

    counting_enabled = false;
    tracking_deadline = sim.T_max;
    track_steps_after_trigger = 100;  % 外源触发后额外追踪步数

    % 先运行稳定期，不计入统计
    for step = 1:sim.stabilization_steps
        sim.step();
    end

    max_remaining_steps = max(1, sim.T_max - sim.current_step);

    for step = 1:max_remaining_steps
        prev_active = sim.isActive;
        sim.step();

        % 外源脉冲第一次触发时，开启统计并将种子标记为世代 0
        if ~counting_enabled && sim.external_pulse_triggered
            counting_enabled = true;
            tracking_deadline = min(sim.T_max, sim.current_step + track_steps_after_trigger);

            % 外源激活的个体视为世代 0 种子
            seed_indices = sim.getExternallyActivatedIndices();
            if ~isempty(seed_indices)
                seed_indices = seed_indices(:);
                valid_seeds = seed_indices(seed_indices >= 1 & seed_indices <= N);
                if ~isempty(valid_seeds)
                    first_activated(valid_seeds) = true;
                    generation(valid_seeds) = 0;
                end
            end
        end

        newly_active = sim.isActive & ~prev_active;

        if counting_enabled && sim.current_step <= tracking_deadline && any(newly_active)
            activated_indices = find(newly_active);

            % 仅对首次激活的节点进行世代赋值与统计
            valid_new = activated_indices(~first_activated(activated_indices));

            for idx = valid_new'
                parent = sim.src_ids{idx};

                if ~isempty(parent)
                    parent_idx = parent(1);
                    if parent_idx >= 1 && parent_idx <= N
                        parent_idx = round(parent_idx);
                        % 若父节点尚未进入家谱树，则将其视为新的种子（世代 0）
                        if ~first_activated(parent_idx)
                            first_activated(parent_idx) = true;
                            generation(parent_idx) = 0;
                        end

                        g_parent = generation(parent_idx);
                        if isnan(g_parent)
                            g_parent = 0; % 兜底，理论上不应出现
                        end

                        generation(idx) = g_parent + 1;
                        first_activated(idx) = true;
                    else
                        % 父节点索引异常时，将该节点视为新的种子
                        generation(idx) = 0;
                        first_activated(idx) = true;
                    end
                else
                    % 无父节点（例如噪声自发激活），视为新的种子（世代 0）
                    generation(idx) = 0;
                    first_activated(idx) = true;
                end
            end
        end

        if counting_enabled && (~sim.cascade_active || sim.current_step >= tracking_deadline)
            break;
        end
    end

    % 基于世代结构统计分支比
    valid_gen_mask = ~isnan(generation);
    if ~any(valid_gen_mask)
        ratio = 0;
        ratio_partial = 0;
        Z_vec = [];
    else
        gen_values = generation(valid_gen_mask);
        max_gen = max(gen_values);

        % 统计各代事件数 Z_g（g 从 0 到 max_gen）
        Z_vec = zeros(max_gen + 1, 1);
        for g = 0:max_gen
            Z_vec(g + 1) = sum(gen_values == g);
        end

        if max_gen <= 0
            % 只有种子，没有产生后代
            ratio = 0;
            ratio_partial = 0;
        else
            % 全局分支比：跨所有代的 children_total / parents_total
            parents_total = sum(Z_vec(1:end-1));  % 所有父代事件总数
            children_total = sum(Z_vec(2:end));   % 所有子代事件总数

            if parents_total == 0
                ratio = 0;
            else
                ratio = children_total / parents_total;
            end

            % 仅前若干代（0→1→...→K）的分支比：
            % 取局部 K_eff，避免超过本次实验的最大世代数。
            K = max(1, round(max_partial_generations));
            K_eff = min(K, max_gen);

            if K_eff <= 0
                ratio_partial = 0;
            else
                % 父代：Z0 ... Z_{K_eff-1}
                % 子代：Z1 ... Z_{K_eff}
                parents_partial = sum(Z_vec(1:K_eff));
                children_partial = sum(Z_vec(2:K_eff+1));

                if parents_partial == 0
                    ratio_partial = 0;
                else
                    ratio_partial = children_partial / parents_partial;
                end
            end
        end
    end

    % 级联规模：everActivated 的比例
    cascade_size = sum(sim.everActivated) / sim.N;
end

%% 辅助函数：简单 2D 统计（参数 × 重复次数）
function [mean_values, std_values, sem_values] = compute_statistics_2d(raw_data)
%COMPUTE_STATISTICS_2D 对二维数据（param × runs）计算均值 / 标准差 / 标准误

    mean_values = NaN(size(raw_data, 1), 1);
    std_values = mean_values;
    sem_values = mean_values;

    for param_idx = 1:size(raw_data, 1)
        samples = raw_data(param_idx, :);
        samples = samples(~isnan(samples));
        if isempty(samples)
            continue;
        end
        mean_values(param_idx) = mean(samples);
        std_values(param_idx) = std(samples, 0);
        sem_values(param_idx) = std_values(param_idx) / sqrt(numel(samples));
    end
end

%% 辅助函数：默认仿真参数
function params = default_simulation_parameters()
%DEFAULT_SIMULATION_PARAMETERS 返回一套简单默认参数
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

%% 辅助函数：配置并行池
function pool = configure_parallel_pool()
%CONFIGURE_PARALLEL_POOL 配置并行池（与其他实验脚本保持一致风格）
    if ~license('test', 'Distrib_Computing_Toolbox')
        error('需要 Parallel Computing Toolbox 才能运行此脚本。');
    end

    pool = gcp('nocreate');
    if isempty(pool)
        cluster = parcluster('local');
        % 允许最多使用 200 个 worker，但不超过本机可用核心数
        max_workers = min(cluster.NumWorkers, 200);
        if max_workers < 1
            error('没有可用的并行工作线程。');
        end
        pool = parpool(cluster, max_workers);
    end
end
