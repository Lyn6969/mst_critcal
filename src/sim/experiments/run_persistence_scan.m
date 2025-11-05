% run_persistence_scan 持久性扫描实验脚本
%
% 功能概述:
%   1. 遍历运动显著性阈值 (cj_threshold) 与角度噪声强度 (angleNoiseIntensity)
%   2. 基于 ParticleSimulation 模型执行多次仿真
%   3. 计算每组参数下相对质心的空间扩散系数 D_r 以及持久性指标 P = 1/sqrt(D_r)
%   4. 汇总统计结果并写入 data/experiments/persistence_scan/<时间戳>/ 目录
%
% 持久性扫描实验的目的：
%   - 研究粒子群在不同参数条件下的运动持久性特征
%   - 分析运动显著性阈值和角度噪声对群体空间凝聚力的影响
%   - 通过相对质心扩散系数 D_r 量化群体相对分散速率
%   - 通过持久性指标 P = 1/sqrt(D_r) 评估群体保持结构的能力
%
% 使用说明:
%   直接运行本脚本。可根据需要调整下方的参数设置与扫描范围。
%
% 作者：李亚男
% 日期：2025年
% MATLAB 2025a 兼容

clc;            % 清除命令行窗口内容
clear;          % 清除工作空间变量
close all;      % 关闭所有图形窗口
addpath(genpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', '..')));

%% 1. 全局参数设置 ---------------------------------------------------------
fprintf('=================================================\n');
fprintf('     持久性扫描实验 (无领航者)\n');
fprintf('=================================================\n\n');

% 实验配置参数结构体
config = struct();
config.num_runs_per_setting = 50;          % 每组参数的重复次数，增加可提高统计可靠性
config.desired_workers = 100;               % 并行工作进程数量，为空则沿用现有并行池设置
config.burn_in_ratio = 0.5;                % 拟合扩散系数时丢弃的前期比例，避免初始瞬态影响
config.min_diffusion = 1e-3;               % 扩散系数的下限，防止数值异常
config.min_fit_points = 30;                % 线性拟合所需的最少数据点数

% 粒子仿真基础参数，可按项目需求调整
base_params = struct();
base_params.N = 200;                       % 粒子数量，影响群体行为复杂度
base_params.rho = 1;                       % 粒子密度参数
base_params.v0 = 1;                        % 粒子基础速度
base_params.angleUpdateParameter = 10;     % 角度更新参数，影响方向调整速度
base_params.angleNoiseIntensity = 0.1;     % 角度噪声强度，将被扫描参数覆盖
base_params.T_max = 400;                   % 最大仿真步数，影响数据采集长度
base_params.dt = 0.1;                      % 时间步长，影响仿真精度和计算量
base_params.radius = 5;                    % 粒子相互作用半径
base_params.deac_threshold = 0.1745;       % 失活阈值(弧度)，约10度
base_params.cj_threshold = 1.0;            % 运动显著性阈值，将被扫描参数覆盖
base_params.fieldSize = 50;                % 仿真场域大小
base_params.initDirection = pi/4;          % 初始方向(45度)
base_params.useFixedField = true;          % 是否使用固定边界场域

% 参数扫描范围设置 (可按需调整步长与范围)
cj_thresholds = 0.0:0.1:5.0;               % 运动显著性阈值扫描范围：从0到6，步长0.1
noise_levels = 0.0:0.0025:0.125;             % 角度噪声强度扫描范围：从0到1，步长0.01

% 显示扫描参数信息
fprintf('扫描的 cj_threshold 个数: %d\n', numel(cj_thresholds));
fprintf('扫描的噪声水平个数: %d\n', numel(noise_levels));
fprintf('每组参数重复次数: %d\n', config.num_runs_per_setting);

% 随机数种子基线，方便结果重现
base_seed = 20250125;                      % 基础随机种子，确保实验可重复性

%% 2. 输出目录准备 ---------------------------------------------------------
% 生成基于当前时间戳的唯一输出目录，避免结果覆盖
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
output_root = fullfile('data', 'experiments', 'persistence_scan', timestamp);
fig_dir = fullfile(output_root, 'figures');  % 图形输出子目录

% 创建输出目录结构，如果不存在则新建
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end
fprintf('结果将写入: %s\n\n', output_root);

%% 3. 结果缓存变量初始化 ----------------------------------------------------
% 计算扫描参数组合的总数
num_cj = numel(cj_thresholds);              % cj_threshold 参数点数量
num_noise = numel(noise_levels);            % 噪声水平参数点数量
total_tasks = num_cj * num_noise;           % 总任务数(参数组合数)

% 进度跟踪器初始化
progress_update = create_progress_tracker(total_tasks);

% 预分配结果存储数组，使用NaN初始化便于后续数据完整性检查
% 注意：使用线性数组存储，便于并行计算，后续会重塑为矩阵形式
D_mean_linear = NaN(total_tasks, 1);        % 扩散系数均值线性数组
D_std_linear = NaN(total_tasks, 1);         % 扩散系数标准差线性数组
P_std_linear = NaN(total_tasks, 1);         % 持久性指标标准差线性数组
raw_D_linear = cell(total_tasks, 1);        % 原始扩散系数数据元胞数组
raw_P_linear = cell(total_tasks, 1);        % 原始持久性指标数据元胞数组
                              % 初始化并行池句柄
% 配置并行计算池，根据系统资源和用户需求设置工作进程数
pool = configure_parallel_pool(config.desired_workers);
fprintf('并行模式启用: %d workers\n\n', pool.NumWorkers);

%% 4. 主循环: 遍历阈值与噪声 ------------------------------------------------
loop_timer = tic;                          % 启动计时器，记录总执行时间

progress_queue = parallel.pool.DataQueue;
afterEach(progress_queue, @(~) progress_update());

% 并行执行：使用 parfor 处理所有参数组合
parfor task_idx = 1:total_tasks
    % 将线性任务索引转换为二维参数索引
    [noise_idx, cj_idx] = ind2sub([num_noise, num_cj], task_idx);
    cj_value = cj_thresholds(cj_idx);      % 当前运动显著性阈值
    noise_value = noise_levels(noise_idx); % 当前噪声水平

    % 对当前参数组合执行多次仿真并计算统计量
    [D_runs, P_runs] = evaluate_single_setting( ...
        base_params, cj_value, noise_value, config, base_seed);

    % 存储原始数据和统计结果
    raw_D_linear{task_idx} = D_runs;       % 原始扩散系数数据
    raw_P_linear{task_idx} = P_runs;       % 原始持久性指标数据
    D_mean_linear(task_idx) = mean(D_runs, 'omitnan');   % 扩散系数均值
    D_std_linear(task_idx) = std(D_runs, 0, 'omitnan');  % 扩散系数标准差
    P_std_linear(task_idx) = std(P_runs, 0, 'omitnan');  % 持久性指标标准差

    send(progress_queue, 1);
end

% 计算并显示总执行时间
elapsed_minutes = toc(loop_timer) / 60;
fprintf('全部扫描完成，总耗时约 %.2f 分钟\n', elapsed_minutes);

% 并行池管理：保留并行池以便后续任务继续复用
% 如需释放可在脚本外手动执行 delete(gcp('nocreate'))
if ~isempty(pool)
    % 保留并行池，不在此处释放
end

% 将线性数组重塑为二维矩阵形式，便于后续分析和可视化
% 矩阵维度：行对应噪声水平，列对应运动显著性阈值
D_mean = reshape(D_mean_linear, [num_noise, num_cj]);  % 扩散系数均值矩阵
D_std = reshape(D_std_linear, [num_noise, num_cj]);    % 扩散系数标准差矩阵
P_mean = 1 ./ sqrt(D_mean);                            % 基于平均扩散计算的持久性矩阵
P_std = reshape(P_std_linear, [num_noise, num_cj]);    % 持久性指标标准差矩阵
raw_D = reshape(raw_D_linear, [num_noise, num_cj]);    % 原始扩散系数数据矩阵
raw_P = reshape(raw_P_linear, [num_noise, num_cj]);    % 原始持久性指标数据矩阵

%% 5. 结果保存 --------------------------------------------------------------
% 创建结果结构体，包含所有实验数据和元数据
results = struct();
results.description = 'Persistence scan without external leaders';  % 实验描述
results.timestamp = timestamp;                                        % 时间戳
results.generated_at = datetime('now');                              % 生成时间
results.base_params = base_params;                                  % 基础仿真参数
results.config = config;                                            % 实验配置
results.cj_thresholds = cj_thresholds;                              % 运动显著性阈值序列
results.noise_levels = noise_levels;                                % 噪声水平序列
results.D_mean = D_mean;                                            % 扩散系数均值矩阵
results.D_std = D_std;                                              % 扩散系数标准差矩阵
results.P_mean = P_mean;                                            % 持久性指标均值矩阵
results.P_std = P_std;                                              % 持久性指标标准差矩阵
results.raw_D = raw_D;                                              % 原始扩散系数数据
results.raw_P = raw_P;                                              % 原始持久性指标数据

% 保存结果到MAT文件，使用-v7.3格式支持大文件和压缩
output_mat = fullfile(output_root, 'persistence_results.mat');
save(output_mat, 'results', '-v7.3');
fprintf('统计结果已保存至: %s\n', output_mat);

%% 6. 快速可视化 ------------------------------------------------------------
% 创建持久性热力图，直观展示参数对群体行为的影响
persist_fig = figure('Name', '持久性热力图', ...
    'Position', [100, 100, 1200, 500], 'Color', 'white');

% 左侧子图：持久性指标热力图
subplot(1, 2, 1);
imagesc(cj_thresholds, noise_levels, P_mean);  % 显示持久性均值矩阵
set(gca, 'YDir', 'normal');                    % 设置Y轴正向显示
colorbar;                                      % 显示颜色条
xlabel('cj\_threshold');                       % X轴标签
ylabel('角度噪声强度');                        % Y轴标签
title('平均持久性 P (越大表示越稳定)');          % 图标题

% 右侧子图：扩散系数热力图
subplot(1, 2, 2);
imagesc(cj_thresholds, noise_levels, D_mean);  % 显示扩散系数均值矩阵
set(gca, 'YDir', 'normal');                    % 设置Y轴正向显示
colorbar;                                      % 显示颜色条
xlabel('cj\_threshold');                       % X轴标签
ylabel('角度噪声强度');                        % Y轴标签
title('平均相对质心扩散系数 D_r (越小越好)');       % 图标题

% 保存图形文件，支持高分辨率PNG和可编辑FIG格式
heatmap_path = fullfile(fig_dir, 'persistence_heatmap.png');
print(persist_fig, heatmap_path, '-dpng', '-r300');  % 300 DPI PNG格式
savefig(persist_fig, fullfile(fig_dir, 'persistence_heatmap.fig'));  % FIG格式
fprintf('热力图已保存至: %s\n', heatmap_path);

fprintf('\n所有任务完成。\n');

%% ========================================================================
%% 辅助函数
%% ========================================================================

function [D_runs, P_runs] = evaluate_single_setting(base_params, cj_value, noise_value, config, base_seed)
% evaluate_single_setting 对单一参数组合执行多次仿真，计算相对质心扩散系数 D_r 与持久性指标 P
%
% 输入参数:
%   base_params - 基础仿真参数结构体
%   cj_value    - 当前运动显著性阈值
%   noise_value - 当前角度噪声强度
%   config      - 实验配置参数
%   base_seed   - 基础随机种子
%
% 输出参数:
%   D_runs - 多次仿真的相对质心扩散系数结果数组
%   P_runs - 多次仿真的持久性指标结果数组

    num_runs = config.num_runs_per_setting;      % 获取重复次数
    % 预分配结果数组，使用NaN初始化便于后续数据完整性检查
    D_runs = NaN(num_runs, 1);                  % 扩散系数结果数组
    P_runs = NaN(num_runs, 1);                  % 持久性指标结果数组
    
    % 执行多次独立仿真，提高统计可靠性
    for run_idx = 1:num_runs
        % 为每次运行生成唯一随机种子，确保结果可重现
        % 种子计算：基础种子 + 运行索引 + 参数相关的偏移量
        seed = base_seed + run_idx + round(1e3 * cj_value) + round(1e4 * noise_value);
        
        % 执行单次仿真试验
        [D_runs(run_idx), P_runs(run_idx)] = run_single_trial( ...
            base_params, cj_value, noise_value, config, seed);
    end
end

function [D_value, P_value] = run_single_trial(base_params, cj_value, noise_value, config, seed)
% run_single_trial 执行单次仿真并估算相对质心扩散系数及持久性指标
%
% 输入参数:
%   base_params - 基础仿真参数结构体
%   cj_value    - 当前运动显著性阈值
%   noise_value - 当前角度噪声强度
%   config      - 实验配置参数
%   seed        - 随机种子(可选)
%
% 输出参数:
%   D_value - 估算的相对质心扩散系数
%   P_value - 计算的持久性指标 P = 1/sqrt(D_value)

    % 设置随机种子以确保结果可重现
    if nargin >= 5 && ~isempty(seed)
        rng(seed);
    end
    
    % 配置当前仿真的参数
    params = base_params;
    params.cj_threshold = cj_value;             % 设置运动显著性阈值
    params.angleNoiseIntensity = noise_value;   % 设置角度噪声强度
    
    % 读取配置中的拟合与数值下限参数
    if isfield(config, 'min_diffusion') && isnumeric(config.min_diffusion) ...
            && isscalar(config.min_diffusion) && config.min_diffusion > 0
        min_diffusion = config.min_diffusion;
    else
        min_diffusion = 1e-3;
    end
    if isfield(config, 'min_fit_points') && isnumeric(config.min_fit_points) ...
            && isscalar(config.min_fit_points) && config.min_fit_points >= 2
        min_fit_points = max(2, round(config.min_fit_points));
    else
        min_fit_points = 30;
    end
    
    % 创建粒子仿真对象
    simulation = ParticleSimulation(params);
    
    % 获取仿真参数
    T = simulation.T_max;                       % 最大仿真步数
    dt = simulation.dt;                         % 时间步长
    % 计算预热期结束索引，排除初始瞬态影响
    burn_in_index = max(2, floor((T + 1) * config.burn_in_ratio));
    
    % 预分配相对质心的均方位移时间序列
    initial_positions = simulation.positions;
    initial_centroid = mean(initial_positions, 1);
    initial_offsets = initial_positions - initial_centroid;
    msd = zeros(T + 1, 1);
    msd(1) = 0;

    % 执行仿真循环，记录每一步的相对质心均方位移
    for step_idx = 1:T
        simulation.step();  % 执行一步仿真
        positions = simulation.positions;
        centroid = mean(positions, 1);
        centered_pos = positions - centroid;
        rel_disp = centered_pos - initial_offsets;
        squared_disp = sum(rel_disp.^2, 2);
        msd(step_idx + 1) = mean(squared_disp, 'omitnan');
    end
    
    time_vec = (0:T)' * dt;                       % 时间向量
    
    % 提取用于线性拟合的数据段(排除预热期)
    x = time_vec(burn_in_index:end);              % 时间数据
    y = msd(burn_in_index:end);                   % 均方位移数据
    
    % 异常情况处理：数据不足或变化过小
    if numel(x) < max(2, min_fit_points) || all(abs(y - y(1)) < eps)
        D_value = NaN;
    else
        x_shift = x - x(1);
        y_shift = y - y(1);
        if any(x_shift > 0) && any(abs(y_shift) > eps)
            smooth_window = max(5, floor(numel(y_shift) * 0.1));
            if smooth_window > 1
                y_shift = smoothdata(y_shift, 'movmean', smooth_window);
            end
            slope = lsqnonneg(x_shift(:), y_shift(:));
            if slope <= 0
                D_value = NaN;
            else
                D_value = slope;
            end
        else
            D_value = NaN;
        end
    end
    
    % 计算持久性指标：P = 1/sqrt(D_r)
    % 持久性指标越大，表示群体结构越稳定
    if isnan(D_value)
        P_value = NaN;
    else
        D_value = max(D_value, min_diffusion);
        P_value = 1 / sqrt(D_value);
    end
end

function pool = configure_parallel_pool(desired_workers)
% configure_parallel_pool 复用项目现有的并行池配置策略，保持一致性。
%
% 输入参数:
%   desired_workers - 期望的并行工作进程数(可选，为空则使用默认值)
%
% 输出参数:
%   pool - 配置好的并行池对象
%
% 功能说明:
%   - 检查Parallel Computing Toolbox许可证
%   - 智能配置并行工作进程数量
%   - 复用现有并行池或创建新的并行池
%
% 设计原则:
%   - 避免频繁创建和销毁并行池，提高效率
%   - 根据系统资源自动调整并行进程数
%   - 提供清晰的错误信息

    % 参数默认值处理
    if nargin < 1
        desired_workers = [];
    end

    % 检查并行计算工具箱许可证
    if ~license('test', 'Distrib_Computing_Toolbox')
        error('需要 Parallel Computing Toolbox 才能运行并行实验。');
    end

    % 获取本地集群信息和最大可用工作进程数
    cluster = parcluster('local');
    max_workers = min(cluster.NumWorkers, 180);  % 限制最大工作进程数，防止资源过载
    if max_workers < 1
        error('当前环境未检测到可用的并行 worker。');
    end

    % 检查是否已有运行的并行池
    pool = gcp('nocreate');
    if isempty(pool)
        % 创建新的并行池
        workers = select_worker_count(desired_workers, max_workers);
        pool = parpool(cluster, workers);
        return;
    end

    % 如果已有并行池，检查是否需要调整工作进程数
    if ~isempty(desired_workers)
        target = select_worker_count(desired_workers, max_workers);
        if pool.NumWorkers ~= target
            % 关闭当前并行池并创建新的
            delete(pool);
            pool = parpool(cluster, target);
        end
    end
end

function progress_handle = create_progress_tracker(total_tasks)
% create_progress_tracker 返回一个函数句柄，每调用一次即更新整体进度。
%
% 输入:
%   total_tasks - 需要完成的任务总数
%
% 输出:
%   progress_handle - 无参函数句柄，执行时刷新进度显示

    if total_tasks <= 0
        progress_handle = @() [];
        return;
    end
    
    count = 0;
    last_print = tic;
    start_time = tic;
    min_interval = 0.5;
    
    fprintf('进度: 0/%d (0.0%%)', total_tasks);
    
    progress_handle = @update_progress;
    
    function update_progress()
        count = count + 1;
        if toc(last_print) < min_interval && count < total_tasks
            return;
        end
        pct = count / total_tasks * 100;
        elapsed = toc(start_time);
        if count > 0 && elapsed > 0
            remaining = max(total_tasks - count, 0);
            eta = (elapsed / count) * remaining;
            eta_text = sprintf(' ETA %.1fs', eta);
        else
            eta_text = '';
        end
        fprintf('\r进度: %d/%d (%.1f%%)%s', count, total_tasks, pct, eta_text);
        last_print = tic;
        if count >= total_tasks
            fprintf('\n');
        end
    end
end

function workers = select_worker_count(requested, max_workers)
% select_worker_count 对用户请求的 worker 数做安全裁剪
%
% 输入参数:
%   requested   - 用户请求的工作进程数
%   max_workers - 系统支持的最大工作进程数
%
% 输出参数:
%   workers - 经过安全裁剪后的工作进程数
%
% 功能说明:
%   - 确保工作进程数在合理范围内
%   - 处理空请求情况
%   - 防止请求超过系统容量
%
% 安全策略:
%   - 最少保证1个工作进程
%   - 最多不超过系统支持的最大值
%   - 优先使用系统最大值以获得最佳性能

    if isempty(requested)
        % 用户未指定，使用系统最大值
        workers = max_workers;
    else
        % 确保至少有1个工作进程，且不超过系统最大值
        workers = max(1, min(requested, max_workers));
    end
end
