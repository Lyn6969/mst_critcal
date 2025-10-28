% run_cj_tradeoff_scan
% =========================================================================
% 目的：
%   - 扫描运动显著性阈值 cj_threshold ∈ [0, 5]，观察其对集群响应性 R 与
%     持久性 P 的影响，从而直观展示两者之间的权衡关系。
%   - 每个参数点重复 50 次实验，统计均值与标准误差，并输出权衡图。
%
% 数据指标：
%   - 响应性 R：复用外源脉冲实验，按 high order 定义进行投影积分。
%   - 持久性 P：在无外源脉冲条件下测量相对质心扩散，P = 1/sqrt(D)。
%
% 输出：
%   - 控制台进度与汇总信息
%   - results/tradeoff/ 目录下的 MAT 数据文件
%   - 带误差棒的 R-P 权衡散点图（颜色编码参数值）
%
% 使用方式：
%   直接运行脚本，可视情况调整下方的参数配置。
% =========================================================================

% 清理工作空间环境
clc;            % 清空命令行窗口，确保输出信息清晰
clear;          % 清空工作空间变量，避免变量冲突
close all;      % 关闭所有图形窗口，释放图形资源

%% 1. 全局参数配置 ---------------------------------------------------------------
fprintf('=================================================\n');
fprintf('   运动显著性阈值扫描：响应性 vs 持久性\n');
fprintf('=================================================\n\n');

% 公共仿真参数（ParticleSimulation / ParticleSimulationWithExternalPulse 共用字段）
base_common = struct();
base_common.N = 200;                      % 粒子数量：群体中的粒子总数
base_common.rho = 1;                      % 密度参数：影响粒子间的相互作用强度
base_common.v0 = 1;                       % 粒子基础速度：每个粒子的标准运动速度
base_common.angleUpdateParameter = 10;   % 角度更新参数：控制粒子方向变化的响应速度
base_common.angleNoiseIntensity = 0.05;  % 角度噪声强度（适度噪声，才能观察权衡趋势）
base_common.T_max = 600;                 % 最大仿真时间步数：总仿真时长
base_common.dt = 0.1;                    % 时间步长：每步仿真的时间间隔
base_common.radius = 5;                  % 交互半径：粒子间相互影响的最大距离
base_common.deac_threshold = 0.1745;     % 失活阈值（弧度）：粒子失活的临界角度
base_common.cj_threshold = 1.0;          % 运动显著性阈值（将在循环中覆盖）
base_common.fieldSize = 50;              % 场地大小：仿真空间的边长
base_common.initDirection = pi/4;        % 初始方向：粒子的初始运动方向（45度）
base_common.useFixedField = true;        % 是否使用固定边界：true表示使用固定边界条件

% 响应性专用参数（继承公共参数并添加特定设置）
resp_params = base_common;
resp_params.stabilization_steps = 200;   % 稳定化步数：等待系统达到平衡状态的步数
resp_params.forced_turn_duration = 200;  % 强制转向持续时间：外源脉冲作用的时间长度

% 持久性评估专用参数（无需外源设置）
pers_params = base_common;
pers_params.T_max = 800;                 % 更长时间窗口：延长仿真时间以获得更稳定的扩散系数估计

% 参数扫描与重复次数设置
cj_thresholds = 0.0:0.1:5.0;            % cj_threshold 扫描范围：从0到5，步长0.1，覆盖低到高的显著性阈值
num_params = numel(cj_thresholds);       % 参数点总数：扫描范围内的参数点数量
num_runs = 50;                           % 每个参数点的重复实验次数：确保统计结果的可靠性
num_angles = 1;                          % 投影角度数量：只取领导方向进行投影分析

% 持久性拟合配置参数
pers_cfg = struct();
pers_cfg.burn_in_ratio = 0.25;          % 预热期比例：前25%的数据不用于拟合，避免初始瞬态影响
pers_cfg.min_diffusion = 1e-4;           % 最小扩散系数阈值：防止数值计算中的除零问题
pers_cfg.min_fit_points = 40;            % 最小拟合点数：确保拟合过程有足够的数据点支持

% 结果容器预分配（遵循MATLAB性能优化原则：内存预分配）
time_vec = (0:resp_params.T_max)' * resp_params.dt;  % 时间向量：用于响应性计算的时间轴
R_raw = NaN(num_params, num_runs);       % 响应性原始数据矩阵：存储每次实验的响应性测量值
P_raw = NaN(num_params, num_runs);       % 持久性原始数据矩阵：存储每次实验的持久性测量值
trigger_failures = zeros(num_params, 1); % 触发失败次数统计：记录外源脉冲未成功触发的次数
diffusion_values = NaN(num_params, num_runs); % 扩散系数原始数据：存储每次实验拟合得到的扩散系数

% 结果输出目录设置
results_dir = fullfile('results', 'tradeoff');  % 结果保存路径：创建tradeoff子目录存储权衡分析结果
if ~exist(results_dir, 'dir')            % 检查目录是否存在，不存在则创建
    mkdir(results_dir);
end
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));  % 生成时间戳：确保文件名唯一性
output_mat = fullfile(results_dir, sprintf('cj_tradeoff_%s.mat', timestamp));  % 数据文件路径：保存实验数据
output_fig = fullfile(results_dir, sprintf('cj_tradeoff_%s.png', timestamp));  % 图像文件路径：保存权衡关系图

% 显示实验配置信息
fprintf('扫描参数点: %d，重复次数: %d，噪声强度: %.3f\n\n', ...
    num_params, num_runs, base_common.angleNoiseIntensity);

%% 2. 主循环 ---------------------------------------------------------------------
base_seed = 20250301;                    % 基础随机种子：确保实验可重复性
loop_tic = tic;                          % 开始计时整个实验过程：用于统计总实验时间

% 并行配置
config = struct();
config.desired_workers = [];
config.progress_interval = 5;
total_tasks = num_params * num_runs;

progress_queue = parallel.pool.DataQueue;
configure_parallel_pool(config.desired_workers);
pool = gcp();
fprintf('并行模式: %d workers\n', pool.NumWorkers);

update_progress('init', total_tasks, loop_tic, config.progress_interval);
afterEach(progress_queue, @(~) update_progress('step'));

for param_idx = 1:num_params
    current_cj = cj_thresholds(param_idx);
    resp_params.cj_threshold = current_cj;
    pers_params.cj_threshold = current_cj;

    fprintf('参数 %02d/%02d: cj_threshold = %.2f\n', param_idx, num_params, current_cj);
    param_tic = tic;

    local_R = NaN(1, num_runs);
    local_P = NaN(1, num_runs);
    local_D = NaN(1, num_runs);
    local_fail = false(1, num_runs);

    parfor run_idx = 1:num_runs
        seed_base = base_seed + (param_idx - 1) * num_runs + run_idx;

        [R_value, triggered] = run_single_responsiveness_trial(resp_params, num_angles, time_vec, seed_base);
        [P_value, D_value] = run_single_persistence_trial(pers_params, pers_cfg, seed_base + 10000);

        local_fail(run_idx) = (~triggered || isnan(R_value));

        local_R(run_idx) = R_value;
        local_P(run_idx) = P_value;
        local_D(run_idx) = D_value;

        send(progress_queue, 1);
    end

    R_raw(param_idx, :) = local_R;
    P_raw(param_idx, :) = local_P;
    diffusion_values(param_idx, :) = local_D;
    trigger_failures(param_idx) = sum(local_fail);

    fprintf('    完成。触发失败 %d 次，用时 %.1fs\n', ...
        trigger_failures(param_idx), toc(param_tic));
end

total_minutes = toc(loop_tic) / 60;
fprintf('\n全部实验完成，总耗时 %.2f 分钟\n\n', total_minutes);

%% 3. 统计分析 -------------------------------------------------------------------
% 对原始数据进行统计分析，计算均值、标准差和标准误差

% 响应性统计
R_mean = mean(R_raw, 2, 'omitnan');      % 计算每行（每个参数点）的均值，忽略NaN值
R_std = std(R_raw, 0, 2, 'omitnan');     % 计算样本标准差（N-1分母），衡量数据离散程度
R_sem = R_std ./ sqrt(num_runs);         % 计算标准误差（标准差/√N），用于估计均值的不确定性

% 持久性统计
P_mean = mean(P_raw, 2, 'omitnan');      % 持久性均值：每个cj_threshold下的平均持久性
P_std = std(P_raw, 0, 2, 'omitnan');     % 持久性标准差：持久性数据的离散程度
P_sem = P_std ./ sqrt(num_runs);         % 持久性标准误差：持久性均值估计的不确定性

% 扩散系数统计
D_mean = mean(diffusion_values, 2, 'omitnan');  % 扩散系数均值：每个参数下的平均扩散系数

% 持久性全局归一化
valid_P = P_raw(~isnan(P_raw));
if isempty(valid_P) || range(valid_P) < eps
    warning('持久性数据无法归一化，全部置零');
    P_norm_raw = zeros(size(P_raw));
else
    P_min = min(valid_P);
    P_range = max(valid_P) - P_min;
    P_norm_raw = (P_raw - P_min) / P_range;
end
P_norm_mean = mean(P_norm_raw, 2, 'omitnan');
P_norm_std = std(P_norm_raw, 0, 2, 'omitnan');
P_norm_sem = P_norm_std ./ sqrt(num_runs);

%% 4. 绘制权衡图 -----------------------------------------------------------------
% 创建响应性-持久性权衡关系图
figure('Name', 'cj 阈值调节响应性-持久性权衡', 'Color', 'white', 'Position', [120, 120, 900, 600]);
hold on;                                 % 保持当前图形，允许叠加绘制

% 绘制散点图：x轴为响应性，y轴为（归一化）持久性，颜色编码cj_threshold值
scatter_handle = scatter(R_mean, P_norm_mean, 70, cj_thresholds, 'filled');
colormap(parula);                        % 使用parula颜色映射：从蓝到红的渐变色
cb = colorbar;                           % 添加颜色条：显示cj_threshold与颜色的对应关系
cb.Label.String = 'cj\_threshold';       % 颜色条标签：说明颜色代表的参数

% 为每个数据点添加误差棒
for idx = 1:num_params
    if isnan(R_mean(idx)) || isnan(P_norm_mean(idx))
        continue;                        % 跳过无效数据点
    end
    % 绘制误差棒：水平和垂直方向分别表示响应性和持久性的标准误差
    errorbar(R_mean(idx), P_norm_mean(idx), P_norm_sem(idx), P_norm_sem(idx), R_sem(idx), R_sem(idx), ...
        'Color', [0.35 0.35 0.35], 'LineWidth', 0.9, 'CapSize', 6);
end

% 绘制趋势线（连接各数据点）
plot(R_mean, P_norm_mean, '-', 'Color', [0.25 0.45 0.8], 'LineWidth', 1.2);

% 设置图形标签和标题
xlabel('响应性 R');                      % x轴标签：群体对外源脉冲的响应能力
ylabel('归一化持久性 \hat{P}');        % y轴标签：归一化后的持久性
title('运动显著性阈值下的响应性-持久性权衡');  % 图形标题：展示核心研究问题
grid on;                                 % 显示网格：便于读取数值
set(gca, 'FontSize', 11);                % 设置坐标轴字体大小：提高可读性

% 添加关键参数点的文本标注
text(R_mean(1), P_norm_mean(1), '  cj=0.0', 'Color', [0.2 0.2 0.2]);  % 起始点：最低阈值
text(R_mean(end), P_norm_mean(end), sprintf('  cj=%.1f', cj_thresholds(end)), ...
    'Color', [0.2 0.2 0.2]);  % 结束点：最高阈值

% 保存图形
saveas(gcf, output_fig);
fprintf('图像已保存至: %s\n', output_fig);

%% 5. 保存数据 --------------------------------------------------------------------
% 创建结果结构体，包含所有实验数据和参数
results = struct();
results.description = 'cj threshold trade-off scan';  % 实验描述
results.timestamp = timestamp;                       % 时间戳
results.parameters = struct('resp', resp_params, 'pers', pers_params);  % 参数结构
results.cj_thresholds = cj_thresholds;               % cj_threshold数组
results.num_runs = num_runs;                         % 重复次数
results.num_angles = num_angles;                     % 投影角度数
results.base_seed = base_seed;                       % 基础随机种子
results.trigger_failures = trigger_failures;          % 触发失败统计
results.R_raw = R_raw;                              % 响应性原始数据
results.P_raw = P_raw;                              % 持久性原始数据
results.D_raw = diffusion_values;                   % 扩散系数原始数据
results.R_mean = R_mean;                            % 响应性均值
results.R_std = R_std;                              % 响应性标准差
results.R_sem = R_sem;                              % 响应性标准误差
results.P_mean = P_mean;                            % 持久性均值
results.P_std = P_std;                              % 持久性标准差
results.P_sem = P_sem;                              % 持久性标准误差
results.P_norm_mean = P_norm_mean;                  % 归一化持久性均值
results.P_norm_std = P_norm_std;                    % 归一化持久性标准差
results.P_norm_sem = P_norm_sem;                    % 归一化持久性标准误差
results.D_mean = D_mean;                            % 扩散系数均值
results.total_minutes = total_minutes;              % 总实验时间
results.config = config;
results.matlab_version = version;                   % MATLAB版本信息
results.P_norm_raw = P_norm_raw;

% 保存为MAT文件（使用-v7.3格式支持大文件）
save(output_mat, 'results', '-v7.3');
fprintf('数据已保存至: %s\n', output_mat);

%% ========================================================================
%                           辅助函数定义
% ========================================================================

function [R_value, triggered] = run_single_responsiveness_trial(params, num_angles, time_vec, seed)
    % 运行单次响应性试验：测量群体对外源脉冲的响应能力
    % 输入：
    %   params - 仿真参数结构体：包含所有仿真所需的参数
    %   num_angles - 投影角度数量：用于多角度分析群体响应
    %   time_vec - 时间向量：用于积分计算的时间轴
    %   seed - 随机种子：确保实验可重复性
    % 输出：
    %   R_value - 响应性指标值：量化群体对外源脉冲的响应强度
    %   triggered - 是否成功触发外源脉冲：标志外源脉冲是否被成功触发
    
    rng(seed);  % 设置随机种子确保可重复性
    
    % 创建带外源脉冲的粒子仿真对象
    sim = ParticleSimulationWithExternalPulse(params);
    sim.external_pulse_count = 1;           % 设置外源脉冲数量：只施加一次外源脉冲
    sim.resetCascadeTracking();             % 重置级联跟踪：清除之前的级联状态
    sim.initializeParticles();              % 初始化粒子位置和方向：设置初始状态

    % 预分配速度历史记录矩阵（性能优化：内存预分配）
    V_history = zeros(params.T_max + 1, 2);  % 速度历史：[x方向速度, y方向速度]
    V_history(1, :) = compute_average_velocity(sim.theta, params.v0);  % 记录初始速度

    % 预分配投影历史记录矩阵
    projection_history = zeros(params.T_max + 1, num_angles);
    triggered = false;                      % 脉冲触发标志：初始设为未触发
    n_vectors = [];                         % 投影方向向量：用于计算速度投影
    t_start = NaN;                          % 脉冲触发时间：记录外源脉冲开始时间

    % 主仿真循环
    for t = 1:params.T_max
        sim.step();                         % 执行一步仿真：更新粒子状态
        V_history(t + 1, :) = compute_average_velocity(sim.theta, params.v0);  % 记录平均速度

        % 检查外源脉冲是否被触发（且之前未被触发）
        if ~triggered && sim.external_pulse_triggered
            triggered = true;               % 设置触发标志：标记脉冲已触发
            t_start = t;                    % 记录触发时间：用于后续积分计算
            
            % 找到被外源激活的粒子（领导者）
            leader_idx = find(sim.isExternallyActivated, 1, 'first');
            if isempty(leader_idx)
                leader_idx = 1;             % 如果没找到，使用第一个粒子
            end
            
            % 获取目标方向
            target_theta = sim.external_target_theta(leader_idx);
            
            % 计算投影方向向量
            if num_angles <= 1
                phi_list = target_theta;    % 单角度：直接使用目标方向
            else
                phi_offsets = linspace(0, pi, num_angles);  % 多角度：均匀分布在0到π之间
                phi_list = target_theta + phi_offsets;
            end
            n_vectors = [cos(phi_list); sin(phi_list)];  % 构造投影向量矩阵
        end

        % 如果脉冲已触发，计算速度在投影方向上的分量
        if triggered
            projection_history(t + 1, :) = V_history(t + 1, :) * n_vectors;
        end
    end

    % 如果脉冲未被触发，返回NaN
    if ~triggered || isnan(t_start)
        R_value = NaN;
        return;
    end

    % 计算响应性指标：在脉冲作用窗口内的投影积分
    v0 = params.v0;                        % 粒子速度：用于归一化
    T_window = params.forced_turn_duration; % 脉冲作用时间窗口：外源脉冲的持续时间
    t_end = min(t_start + T_window, params.T_max);  % 结束时间：确保不超出仿真范围

    % 为每个投影角度计算响应性
    r_history = NaN(num_angles, 1);
    for angle_idx = 1:num_angles
        proj = projection_history(:, angle_idx);  % 获取该角度的投影历史
        % 使用梯形法计算投影积分
        integral_value = trapz(time_vec(t_start+1:t_end+1), proj(t_start+1:t_end+1));
        duration = time_vec(t_end+1) - time_vec(t_start+1);  % 积分时间长度
        if duration > 0
            % 归一化响应性：积分值除以(v0 * 时间)
            r_history(angle_idx) = integral_value / (v0 * duration);
        end
    end

    % 返回所有角度响应性的平均值
    R_value = mean(r_history(~isnan(r_history)));
end

function [P_value, D_value] = run_single_persistence_trial(params, cfg, seed)
    % 运行单次持久性试验：测量群体运动的稳定性
    % 输入：
    %   params - 仿真参数结构体：包含所有仿真所需的参数
    %   cfg - 持久性拟合配置：包含拟合相关的配置参数
    %   seed - 随机种子：确保实验可重复性
    % 输出：
    %   P_value - 持久性指标值 (P = 1/√D)：量化群体运动的稳定性
    %   D_value - 扩散系数值：描述群体扩散的速率
    
    rng(seed);  % 设置随机种子
    
    % 创建基础粒子仿真对象（无外源脉冲）
    sim = ParticleSimulation(params);

    % 获取仿真参数
    T = sim.T_max;                         % 最大时间步数：总仿真步数
    dt = sim.dt;                           % 时间步长：每步的时间间隔
    burn_in_index = max(2, floor((T + 1) * cfg.burn_in_ratio));  % 预热期结束索引

    % 记录初始位置和质心
    initial_positions = sim.positions;     % 初始粒子位置：所有粒子的起始坐标
    centroid0 = mean(initial_positions, 1); % 初始质心位置：粒子群的重心位置
    offsets0 = initial_positions - centroid0; % 粒子相对于初始质心的偏移

    % 预分配均方位移（MSD）数组（性能优化）
    msd = zeros(T + 1, 1);
    msd(1) = 0;                           % 初始MSD为0：初始时刻位移为零

    % 主仿真循环：计算均方位移
    for step_idx = 1:T
        sim.step();                        % 执行一步仿真：更新粒子状态
        positions = sim.positions;         % 当前粒子位置：获取所有粒子的当前坐标
        centroid = mean(positions, 1);     % 当前质心位置：计算当前粒子群的重心
        centered = positions - centroid;    % 相对于当前质心的位置：粒子相对于当前质心的偏移
        rel_disp = centered - offsets0;    % 相对于初始质心偏移的位移：粒子相对于初始位置的净位移
        % 计算均方位移（所有粒子位移平方的平均值）
        msd(step_idx + 1) = mean(sum(rel_disp.^2, 2), 'omitnan');
    end

    % 准备拟合数据：跳过预热期
    time_vec = (0:T)' * dt;               % 时间向量：创建时间轴
    x = time_vec(burn_in_index:end);      % 用于拟合的时间数据：跳过预热期的数据
    y = msd(burn_in_index:end);           % 用于拟合的MSD数据：跳过预热期的MSD

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
            
            % 非负最小二乘拟合：MSD = 4*D*t（二维扩散）
            slope = lsqnonneg(x_shift(:), y_shift(:));
            if slope <= 0
                D_value = NaN;             % 拟合结果无效
            else
                D_value = slope / 4;       % 扩散系数（二维情况下MSD = 4Dt）
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

function V = compute_average_velocity(theta, v0)
    % 计算粒子群体的平均速度向量
    % 输入：
    %   theta - 粒子方向角度数组：所有粒子的运动方向角度
    %   v0 - 粒子速度大小：粒子的标准速度值
    % 输出：
    %   V - 平均速度向量 [Vx, Vy]：群体在x和y方向的平均速度分量
    
    % 将角度转换为速度向量并计算平均值
    % 使用三角函数将角度转换为x和y方向的速度分量，然后计算平均值
    V = [mean(v0 * cos(theta)), mean(v0 * sin(theta))];
end

function pool = configure_parallel_pool(desired_workers)
    pool = gcp('nocreate');
    if isempty(pool)
        if isempty(desired_workers)
            pool = parpool;
        else
            pool = parpool(desired_workers);
        end
    elseif ~isempty(desired_workers) && pool.NumWorkers ~= desired_workers
        delete(pool);
        pool = parpool(desired_workers);
    end
end

function update_progress(mode, varargin)
    persistent total completed start_timer interval last_tic

    switch mode
        case 'init'
            total_tasks = varargin{1};
            timer_handle = varargin{2};
            interval_minutes = varargin{3};

            total = total_tasks;
            completed = 0;
            start_timer = timer_handle;
            interval = interval_minutes;
            last_tic = tic;
            fprintf('  进度: 0%% (0/%d)\n', total);
        case 'step'
            if isempty(total)
                return;
            end
            completed = completed + 1;
            if toc(last_tic) < interval && completed < total
                return;
            end
            elapsed = toc(start_timer);
            avg_time = elapsed / completed;
            remaining = avg_time * max(total - completed, 0);
            fprintf('  进度: %.1f%% (%d/%d) | 已用 %.1f 分 | 预计剩余 %.1f 分\n', ...
                100 * completed / total, completed, total, elapsed / 60, remaining / 60);
            last_tic = tic;
    end
end
