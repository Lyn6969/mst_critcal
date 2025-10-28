% run_persistence_cj_threshold_scan
% =========================================================================
% 目的：
%   - 在固定噪声水平下扫描运动显著性阈值 cj_threshold，评估集群的持久性。
%   - 参照 run_responsiveness_cj_threshold_scan 的结构，输出均值/标准误差，
%     并将持久性结果归一化到 [0, 1] 便于与响应性对比。
%
% 方法：
%   - 对每个阈值运行多次 Monte Carlo 实验，使用 ParticleSimulation（无外源）。
%   - 通过相对质心的均方位移拟合扩散系数 D_r，转化为持久性指标 P = 1/sqrt(D_r)。
%   - 全局最小-最大归一化，使得 P_norm ∈ [0, 1]。
%
% 输出：
%   - 控制台统计 & 进度日志
%   - 持久性-阈值曲线图 (errorbar)
%   - 结果数据保存至 results/persistence/
% =========================================================================

clc;            % 清空命令行窗口，确保输出信息清晰
clear;          % 清空工作空间变量，避免变量冲突
close all;      % 关闭所有图形窗口，释放图形资源

%% 1. 参数设定 ------------------------------------------------------------------
fprintf('=================================================\n');
fprintf('   持久性扫描 (固定噪声、遍历 cj_threshold)\n');
fprintf('=================================================\n\n');

config = struct();
config.desired_workers = [];              % 期望的并行工作进程数：空表示自动检测
config.progress_interval = 5;              % 进度更新间隔（分钟）：控制进度显示频率

base_params = struct();
base_params.N = 200;                      % 粒子数量：群体中的粒子总数
base_params.rho = 1;                      % 密度参数：影响粒子间的相互作用强度
base_params.v0 = 1;                       % 粒子基础速度：每个粒子的标准运动速度
base_params.angleUpdateParameter = 10;   % 角度更新参数：控制粒子方向变化的响应速度
base_params.angleNoiseIntensity = 0.05;   % 固定噪声：角度噪声强度，保持恒定以观察阈值影响
base_params.T_max = 800;                  % 最大仿真时间步数：总仿真时长
base_params.dt = 0.1;                    % 时间步长：每步仿真的时间间隔
base_params.radius = 5;                  % 交互半径：粒子间相互影响的最大距离
base_params.deac_threshold = 0.1745;     % 失活阈值：粒子失活的临界角度（弧度）
base_params.cj_threshold = 1.0;           % 运动显著性阈值：将在循环中覆盖
base_params.fieldSize = 50;               % 场地大小：仿真空间的边长
base_params.initDirection = pi/4;         % 初始方向：粒子的初始运动方向（45度）
base_params.useFixedField = true;         % 是否使用固定边界：true表示使用固定边界条件

cj_thresholds = 0.0:0.1:5.0;             % cj_threshold扫描范围：从0到5，步长0.1
num_params = numel(cj_thresholds);       % 参数点总数：扫描范围内的参数点数量
num_runs = 50;                           % 每个参数点的重复实验次数：确保统计结果的可靠性

fprintf('噪声强度: %.3f\n', base_params.angleNoiseIntensity);
fprintf('阈值范围: [%.1f, %.1f] (步长 %.1f)，参数点 %d 个\n', ...
    cj_thresholds(1), cj_thresholds(end), cj_thresholds(2) - cj_thresholds(1), num_params);
fprintf('每个参数重复次数: %d\n\n', num_runs);

% 持久性估计配置
cfg = struct();
cfg.burn_in_ratio = 0.25;               % 预热期比例：前25%的数据不用于拟合，避免初始瞬态影响
cfg.min_fit_points = 40;                 % 最小拟合点数：确保拟合过程有足够的数据点支持
cfg.min_diffusion = 1e-4;                % 最小扩散系数阈值：防止数值计算中的除零问题

% 输出目录设置
results_dir = fullfile('results', 'persistence');  % 结果保存路径：创建persistence子目录存储持久性分析结果
if ~exist(results_dir, 'dir')            % 检查目录是否存在，不存在则创建
    mkdir(results_dir);
end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));  % 生成时间戳：确保文件名唯一性
output_mat = fullfile(results_dir, sprintf('persistence_cj_scan_%s.mat', timestamp));  % 数据文件路径：保存实验数据
output_fig = fullfile(results_dir, sprintf('persistence_cj_scan_%s.png', timestamp));  % 图像文件路径：保存持久性曲线图

base_seed = 20250320;                   % 基础随机种子：确保实验可重复性
total_timer = tic;                       % 开始计时整个实验过程：用于统计总实验时间

total_tasks = num_params * num_runs;     % 总任务数：参数点数乘以重复次数
progress_queue = parallel.pool.DataQueue; % 并行进度队列：用于在并行计算中传递进度信息
configure_parallel_pool(config.desired_workers);  % 配置并行计算池
pool = gcp();                            % 获取当前并行计算池
fprintf('并行模式: %d workers\n', pool.NumWorkers);  % 显示并行工作进程数

update_progress('init', total_tasks, total_timer, config.progress_interval);  % 初始化进度跟踪
afterEach(progress_queue, @(~) update_progress('step'));  % 设置进度更新回调函数

% 结果容器预分配（遵循MATLAB性能优化原则：内存预分配）
P_raw = NaN(num_params, num_runs);       % 持久性原始数据矩阵：存储每次实验的持久性测量值
D_raw = NaN(num_params, num_runs);       % 扩散系数原始数据矩阵：存储每次实验拟合得到的扩散系数

%% 2. 批量实验 ------------------------------------------------------------------
% 使用并行循环加速实验过程
parfor param_idx = 1:num_params
    current_cj = cj_thresholds(param_idx);  % 当前cj_threshold值：获取当前要测试的显著性阈值
    params_local = base_params;             % 创建本地参数副本：避免并行计算中的变量冲突
    params_local.cj_threshold = current_cj;  % 设置当前参数点的cj_threshold值

    % 本地结果容器（每个并行工作进程独立使用）
    local_P = NaN(1, num_runs);            % 本地持久性结果数组：存储当前参数点的所有重复实验结果
    local_D = NaN(1, num_runs);            % 本地扩散系数结果数组：存储当前参数点的所有扩散系数

    % 内层循环：对当前参数点进行多次重复实验
    for run_idx = 1:num_runs
        seed = base_seed + (param_idx - 1) * num_runs + run_idx;  % 生成唯一随机种子
        [P_val, D_val] = run_single_persistence_trial(params_local, cfg, seed);  % 运行单次持久性试验
        local_P(run_idx) = P_val;           % 存储持久性结果
        local_D(run_idx) = D_val;           % 存储扩散系数
        send(progress_queue, 1);             % 发送进度更新信号
    end

    % 将本地结果写入全局结果矩阵
    P_raw(param_idx, :) = local_P;          % 存储当前参数点的持久性结果
    D_raw(param_idx, :) = local_D;          % 存储当前参数点的扩散系数结果
end

fprintf('\n全部实验完成，总耗时 %.2f 分钟\n\n', toc(total_timer) / 60);  % 显示总实验时间

%% 3. 统计与归一化 ---------------------------------------------------------------
% 对原始数据进行统计分析，计算均值、标准差和标准误差
P_mean = mean(P_raw, 2, 'omitnan');      % 持久性均值：每个cj_threshold下的平均持久性
P_std = std(P_raw, 0, 2, 'omitnan');     % 持久性标准差：持久性数据的离散程度
P_sem = P_std ./ sqrt(num_runs);         % 持久性标准误差：持久性均值估计的不确定性

D_mean = mean(D_raw, 2, 'omitnan');      % 扩散系数均值：每个参数下的平均扩散系数

% 全局 min-max 归一化：将持久性值映射到[0,1]区间便于比较
all_P = P_raw(:);                        % 将持久性矩阵展平为一维数组
valid_P = all_P(~isnan(all_P));          % 提取有效的持久性数据（非NaN）

% 预分配归一化结果容器
P_norm_raw = P_raw;                      % 归一化原始数据：初始化为原始数据
P_norm_mean = NaN(size(P_mean));          % 归一化均值：初始化为NaN
P_norm_std = NaN(size(P_std));           % 归一化标准差：初始化为NaN
P_norm_sem = NaN(size(P_sem));           % 归一化标准误差：初始化为NaN

% 检查数据有效性并进行归一化
if isempty(valid_P) || range(valid_P) < eps
    warning('有效持久性数据过少或数值几乎不变，归一化退化为 0。');
    P_norm_raw(:) = 0;                   % 无效数据时归一化为0
    P_norm_mean(:) = 0;
    P_norm_std(:) = 0;
    P_norm_sem(:) = 0;
else
    P_min = min(valid_P);                 % 最小持久性值
    P_range = max(valid_P) - P_min;       % 持久性值范围
    P_norm_raw = (P_raw - P_min) / P_range;  % Min-max归一化公式：(x-min)/(max-min)

    % 计算归一化后的统计量
    P_norm_mean = mean(P_norm_raw, 2, 'omitnan');  % 归一化持久性均值
    P_norm_std = std(P_norm_raw, 0, 2, 'omitnan'); % 归一化持久性标准差
    P_norm_sem = P_norm_std ./ sqrt(num_runs);     % 归一化持久性标准误差
end

%% 4. 绘图 -----------------------------------------------------------------------
% 创建持久性-阈值关系图
figure('Name', '持久性 vs cj_threshold', 'Color', 'white', 'Position', [120, 120, 900, 540]);
% 绘制带误差棒的线图：显示归一化持久性随cj_threshold的变化趋势
errorbar(cj_thresholds, P_norm_mean, P_norm_sem, 's-', ...
    'LineWidth', 1.4, 'MarkerSize', 5, 'MarkerFaceColor', [0.3 0.6 0.9]);
grid on;                                 % 显示网格：便于读取数值
xlabel('运动显著性阈值 c_j');             % x轴标签：运动显著性阈值
ylabel('归一化持久性 \hat{P}');          % y轴标签：归一化后的持久性指标
title(sprintf('归一化持久性 vs c_j (噪声=%.3f, 每点 %d 次)', ...
    base_params.angleNoiseIntensity, num_runs));  % 图形标题：包含实验参数信息

% 保存图形
saveas(gcf, output_fig);
fprintf('图像已保存至: %s\n', output_fig);

%% 5. 保存数据 --------------------------------------------------------------------
results = struct();
results.description = 'Persistence scan over cj_threshold with fixed noise';
results.timestamp = timestamp;
results.parameters = base_params;
results.cj_thresholds = cj_thresholds;
results.num_runs = num_runs;
results.cfg = cfg;
results.config = config;
results.P_raw = P_raw;
results.D_raw = D_raw;
results.P_mean = P_mean;
results.P_std = P_std;
results.P_sem = P_sem;
results.D_mean = D_mean;
results.P_norm_raw = P_norm_raw;
results.P_norm_mean = P_norm_mean;
results.P_norm_std = P_norm_std;
results.P_norm_sem = P_norm_sem;
results.matlab_version = version;

save(output_mat, 'results', '-v7.3');
fprintf('数据已保存至: %s\n', output_mat);

%% -------------------------------------------------------------------------------
function [P_value, D_value] = run_single_persistence_trial(params, cfg, seed)
    % 运行单次持久性试验：测量群体运动的稳定性
    % 输入：
    %   params - 仿真参数结构体：包含所有仿真所需的参数
    %   cfg - 持久性拟合配置：包含拟合相关的配置参数
    %   seed - 随机种子：确保实验可重复性
    % 输出：
    %   P_value - 持久性指标值：量化群体运动的稳定性
    %   D_value - 扩散系数值：描述群体扩散的速率
    
    rng(seed);  % 设置随机种子确保可重复性
    sim = ParticleSimulation(params);      % 创建基础粒子仿真对象（无外源脉冲）

    % 获取仿真参数
    T = sim.T_max;                       % 最大时间步数：总仿真步数
    dt = sim.dt;                         % 时间步长：每步的时间间隔
    burn_in_index = max(2, floor((T + 1) * cfg.burn_in_ratio));  % 预热期结束索引

    % 记录初始位置和质心
    init_pos = sim.positions;              % 初始粒子位置：所有粒子的起始坐标
    centroid0 = mean(init_pos, 1);        % 初始质心位置：粒子群的重心位置
    offsets0 = init_pos - centroid0;      % 粒子相对于初始质心的偏移
    msd = zeros(T + 1, 1);               % 预分配均方位移数组（性能优化）

    % 主仿真循环：计算均方位移
    for step_idx = 1:T
        sim.step();                       % 执行一步仿真：更新粒子状态
        positions = sim.positions;          % 当前粒子位置：获取所有粒子的当前坐标
        centroid = mean(positions, 1);      % 当前质心位置：计算当前粒子群的重心
        centered = positions - centroid;     % 相对于当前质心的位置：粒子相对于当前质心的偏移
        rel_disp = centered - offsets0;     % 相对于初始质心偏移的位移：粒子相对于初始位置的净位移
        % 计算均方位移（所有粒子位移平方的平均值）
        msd(step_idx + 1) = mean(sum(rel_disp.^2, 2), 'omitnan');
    end

    % 准备拟合数据：跳过预热期
    time_vec = (0:T)' * dt;              % 时间向量：创建时间轴
    x = time_vec(burn_in_index:end);       % 用于拟合的时间数据：跳过预热期的数据
    y = msd(burn_in_index:end);            % 用于拟合的MSD数据：跳过预热期的MSD

    % 检查数据有效性
    if numel(x) < max(2, cfg.min_fit_points) || all(abs(y - y(1)) < eps)
        D_value = NaN;                     % 数据不足或无变化，返回NaN
    else
        % 数据预处理：减去初始值
        x_shift = x - x(1);               % 时间偏移：以起始点为时间零点
        y_shift = y - y(1);               % MSD偏移：以起始点为MSD零点
        
        % 检查数据有效性
        if any(x_shift > 0) && any(abs(y_shift) > eps)
            % 数据平滑：使用移动平均减少噪声
            smooth_window = max(5, floor(numel(y_shift) * 0.1));
            if smooth_window > 1
                y_shift = smoothdata(y_shift, 'movmean', smooth_window);
            end
            
            % 非负最小二乘拟合：MSD = D*t（一维扩散，注意与第一个文件的区别）
            slope = lsqnonneg(x_shift(:), y_shift(:));
            if slope <= 0
                D_value = NaN;             % 拟合结果无效
            else
                D_value = slope;           % 扩散系数（一维情况下MSD = Dt）
            end
        else
            D_value = NaN;                 % 数据无效
        end
    end

    % 计算持久性指标：P = 1/√D
    if isnan(D_value)
        P_value = NaN;
    else
        D_value = max(D_value, cfg.min_diffusion);  % 应用最小扩散系数阈值
        P_value = 1 / sqrt(D_value);       % 持久性与扩散系数成反比
    end
end

function pool = configure_parallel_pool(desired_workers)
    % 配置并行计算池：管理并行计算资源
    % 输入：
    %   desired_workers - 期望的工作进程数，空表示自动检测
    % 输出：
    %   pool - 配置好的并行计算池对象
    
    pool = gcp('nocreate');               % 获取当前并行池（不创建新池）
    if isempty(pool)                       % 如果不存在并行池
        if isempty(desired_workers)         % 如果未指定工作进程数
            pool = parpool;                % 使用默认设置创建并行池
        else
            pool = parpool(desired_workers);  % 使用指定的工作进程数创建并行池
        end
    elseif ~isempty(desired_workers) && pool.NumWorkers ~= desired_workers
        % 如果存在并行池但工作进程数不符合要求
        delete(pool);                      % 删除现有并行池
        pool = parpool(desired_workers);    % 创建新的并行池
    end
end

function update_progress(mode, total_tasks, total_timer_handle, interval_minutes)
    % 进度更新函数：在并行计算中显示实验进度
    % 输入：
    %   mode - 模式：'init'初始化进度跟踪，'step'更新进度
    %   total_tasks - 总任务数：需要完成的实验总数
    %   total_timer_handle - 总计时器句柄：用于计算总耗时
    %   interval_minutes - 更新间隔（分钟）：控制进度显示频率
    
    % 持久变量：在函数调用间保持状态
    persistent total completed start_handle progress_interval last_tic

    switch mode
        case 'init'                        % 初始化模式
            total = total_tasks;             % 设置总任务数
            completed = 0;                  % 初始化已完成任务数
            start_handle = total_timer_handle; % 保存总计时器句柄
            progress_interval = interval_minutes * 60;  % 转换为秒
            last_tic = tic;                 % 开始计时进度更新间隔
            fprintf('  进度: 0%% (0/%d)\n', total);  % 显示初始进度

        case 'step'                         % 进度更新模式
            if isempty(total)                 % 检查是否已初始化
                return;
            end
            completed = completed + 1;       % 增加已完成任务数
            
            % 检查是否需要更新显示（避免过于频繁的更新）
            if toc(last_tic) < progress_interval && completed < total
                return;
            end
            
            % 计算时间和进度信息
            elapsed_seconds = toc(start_handle);  % 已用时间（秒）
            avg_time = elapsed_seconds / completed;  % 平均每任务时间（秒）
            remaining_seconds = avg_time * max(total - completed, 0);  % 预计剩余时间（秒）
            
            % 显示进度信息
            fprintf('  进度: %.1f%% (%d/%d) | 已用 %.1f 分 | 预计剩余 %.1f 分\n', ...
                100 * completed / total, completed, total, ...
                elapsed_seconds / 60, remaining_seconds / 60);
            last_tic = tic;                 % 重置进度更新计时器
    end
end
