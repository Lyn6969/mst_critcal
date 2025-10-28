% visualize_cluster_responsiveness 集群响应性观察脚本
% =========================================================================
% 作用：
%   - 本脚本用于模拟和可视化集群在受到外部单体扰动后的响应行为。
%   - 它复用了 c1 级联实验的场景，即在一个稳定集群中激活单个粒子（领导者）。
%   - 响应性 R 的定义：首先计算群体平均速度 V(t) 在多个不同方向 n_phi 上的投影，
%     并对该投影在扰动后的时间窗口内进行积分，得到 r(phi)。最终的响应性 R 是
%     对所有采样方向的 r(phi) 求平均值。
%   - 提供实时可视化：
%     - 左侧子图：展示粒子群的动态演化过程。
%     - 右上子图：显示平均速度 V(t) 在各个采样方向上的投影随时间的变化。
%     - 右中子图：展示单次运行中，对每个采样方向计算出的响应值 r(phi)。
%     - 右下子图：在所有重复运行结束后，统计并展示最终响应性 R 的分布直方图。
%
% 使用方式：
%   - 直接在 MATLAB 编辑器中运行此脚本。
%   - 程序会提示用户输入“重复次数”和“采样方向个数”。
%     - 重复次数：进行多少次独立的模拟实验。
%     - 采样方向个数：在单次实验中，围绕领导者方向均匀选择多少个角度来计算响应性。
%
% =========================================================================

% --- 环境初始化 ---
clc;            % 清空命令行窗口
clear;          % 清除工作区所有变量
close all;      % 关闭所有图形窗口

%% 1. 输入与参数定义 -----------------------------------------------------------------
fprintf('=== 集群响应性观察 ===\n');

% --- 用户输入 ---
% 获取用户输入的重复运行次数
num_runs = input('请输入重复次数（默认 1 ）: ');
if isempty(num_runs)
    num_runs = 1; % 如果用户直接回车，则使用默认值 1
end

% 获取用户输入的角度采样数量
num_angles = input('请输入采样方向个数（默认 1 ）: ');
if isempty(num_angles)
    num_angles = 1; % 如果用户直接回车，则使用默认值 1
end

fprintf('实验参数：重复 %d 次，每次采样 %d 个方向。\n', num_runs, num_angles);

% --- 模拟参数设置 ---
% 这些参数继承自 delta_c 实验场景，定义了模拟的基本物理属性
base_params = struct();
base_params.N = 200;                        % 粒子数量
base_params.rho = 1;                        % 粒子密度 (此脚本中未直接使用，但为兼容性保留)
base_params.v0 = 1;                         % 粒子速度
base_params.angleUpdateParameter = 10;      % 角度更新参数 (影响粒子间相互作用强度)
base_params.angleNoiseIntensity = 0;        % 角度噪声强度 (设为0表示无噪声)
base_params.T_max = 600;                    % 总模拟步数
base_params.dt = 0.1;                       % 时间步长
base_params.radius = 5;                     % 粒子相互作用半径
base_params.deac_threshold = 0.1745;        % 失活角度阈值 (约 10 度)
base_params.cj_threshold = 0.2;             % 协同判断阈值
base_params.fieldSize = 50;                 % 模拟区域边长
base_params.initDirection = pi/4;           % 粒子初始方向
base_params.useFixedField = true;           % 是否使用固定大小的模拟区域 (无边界)
base_params.stabilization_steps = 200;      % 初始稳定化步数 (让集群达到稳态)
base_params.forced_turn_duration = 200;     % 领导者被强制转向的持续时间

%% 2. 可视化窗口设置 -----------------------------------------------------------------
% 创建一个图形窗口用于显示所有可视化内容
figure('Name', '集群响应性演示', 'Color', 'white', ...
    'Position', [80, 80, 1400, 700]); % 设置窗口位置和大小

% --- 创建子图 ---
% 在窗口中划分不同的区域用于不同的绘图
ax_particles = subplot('Position', [0.05 0.3 0.4 0.65]);   % 左侧：粒子动态图
ax_proj = subplot('Position', [0.5 0.65 0.45 0.25]);      % 右上：速度投影曲线
ax_resp = subplot('Position', [0.5 0.35 0.45 0.25]);      % 右中：单次响应 r(phi) 条形图
ax_stats = subplot('Position', [0.5 0.08 0.45 0.2]);       % 右下：最终响应 R 分布直方图

%% 3. 响应性计算准备 -------------------------------------------------------------
% 生成时间向量，用于后续绘图和积分
time_vec = (0:base_params.T_max)' * base_params.dt;
% 预分配内存，用于存储每次运行计算出的响应性 R 值
R_values = NaN(num_runs, 1);

%% 4. 主循环：进行多次模拟运行 ----------------------------------------------------
for run_idx = 1:num_runs
    % --- 单次运行初始化 ---
    params = base_params; % 复制基础参数，确保每次运行参数一致
    rng(run_idx);  % 设置随机数种子，以保证每次运行的初始状态可复现
    
    % 初始化粒子模拟器对象
    sim = ParticleSimulationWithExternalPulse(params);
    sim.external_pulse_count = 1; % 设置外源脉冲数量为1
    sim.resetCascadeTracking();   % 重置级联跟踪状态
    sim.initializeParticles();    % 根据参数初始化粒子位置和方向

    % 预分配历史数据存储空间
    V_history = zeros(base_params.T_max + 1, 2); % 存储每个时间步的平均速度 (Vx, Vy)
    V_history(1, :) = compute_average_velocity(sim.theta, params.v0); % 计算初始平均速度

    % --- 初始化粒子可视化子图 ---
    subplot(ax_particles);
    cla(ax_particles); % 清空当前子图
    axis(ax_particles, [0 params.fieldSize 0 params.fieldSize]); % 设置坐标轴范围
    axis(ax_particles, 'square'); % 设置为方形坐标轴
    grid(ax_particles, 'on'); % 显示网格
    title(ax_particles, sprintf('运行 %d / %d', run_idx, num_runs));
    xlabel(ax_particles, 'X');
    ylabel(ax_particles, 'Y');
    % 绘制粒子位置，颜色表示其方向
    particles_plot = scatter(sim.positions(:,1), sim.positions(:,2), 36, sim.theta, 'filled');
    hold(ax_particles, 'on');
    % 绘制表示粒子方向的箭头
    arrows_plot = quiver(sim.positions(:,1), sim.positions(:,2), cos(sim.theta), sin(sim.theta), 0.4, ...
        'Color', [0.3 0.3 0.3 0.6], 'LineWidth', 1.0);
    hold(ax_particles, 'off');

    % --- 单次模拟的变量准备 ---
    triggered = false; % 标志位，记录外源脉冲是否已被触发
    projection_history = zeros(base_params.T_max + 1, num_angles); % 存储速度在各方向上的投影历史
    r_history = NaN(num_angles, 1); % 存储当前运行中每个采样方向的响应值 r(phi)
    n_vectors = []; % 存储采样方向的单位向量

    % --- 时间步进循环 ---
    for t = 1:base_params.T_max
        sim.step(); % 执行一步模拟
        V_history(t + 1, :) = compute_average_velocity(sim.theta, params.v0); % 计算并记录当前平均速度

        % --- 检查并处理外源脉冲触发事件 ---
        if ~triggered && sim.external_pulse_triggered
            triggered = true; % 设置触发标志位
            t_start = t;      % 记录触发的起始时间步
            
            % 找到被激活的领导者粒子
            leader_idx = find(sim.isExternallyActivated, 1, 'first');
            if isempty(leader_idx)
                leader_idx = 1; % 如果找不到，默认第一个为领导者
            end
            
            % 获取领导者的目标转向角度
            target_theta = sim.external_target_theta(leader_idx);
            
            % 根据采样数生成用于计算响应性的方向列表
            if num_angles <= 1
                phi_list = target_theta; % 如果只采样1个方向，就用领导者方向
            else
                % 否则，在 [0, pi] 范围内相对于领导者方向均匀采样
                phi_offsets = linspace(0, pi, num_angles);
                phi_list = target_theta + phi_offsets;
            end
            % 将角度列表转换为单位向量
            n_vectors = [cos(phi_list); sin(phi_list)];
        end

        % 如果脉冲已触发，则开始记录速度投影
        if triggered
            % V_history(t + 1, :) 是一个 1x2 向量, n_vectors 是一个 2xnum_angles 矩阵
            % 矩阵乘法得到一个 1xnum_angles 的行向量，包含V在每个n_phi上的投影
            projection_history(t + 1, :) = V_history(t + 1, :) * n_vectors;
        end

        % --- 定期更新可视化 ---
        if mod(t, 5) == 0 % 每 5 步更新一次，以提高性能
            subplot(ax_particles);
            % 更新粒子位置和颜色
            set(particles_plot, 'XData', sim.positions(:,1), 'YData', sim.positions(:,2), 'CData', sim.theta);
            % 更新箭头的位置和方向
            set(arrows_plot, 'XData', sim.positions(:,1), 'YData', sim.positions(:,2), ...
                'UData', cos(sim.theta), 'VData', sin(sim.theta));
            drawnow limitrate; % 刷新图形，limitrate 模式可防止刷新过快
        end
    end

    % --- 单次运行结束，计算响应性 ---
    if ~triggered
        warning('运行 %d 未检测到外源脉冲触发，跳过此轮响应性计算。', run_idx);
        continue; % 如果没有触发脉冲，则无法计算响应性，直接进入下一次运行
    end

    % 定义积分参数
    v0 = params.v0; % 粒子速度
    T_window = params.forced_turn_duration; % 积分时间窗口长度
    t_end = min(t_start + T_window, base_params.T_max); % 确定积分结束时间步

    % 对每个采样方向计算响应值 r(phi)
    for angle_idx = 1:num_angles
        proj = projection_history(:, angle_idx); % 取出在该方向上的投影历史
        % 使用梯形法则计算投影值在时间窗口内的积分
        integral_value = trapz(time_vec(t_start+1:t_end+1), proj(t_start+1:t_end+1));
        duration = time_vec(t_end+1) - time_vec(t_start+1); % 计算积分时长
        
        % 计算归一化的响应值 r(phi)
        if duration <= 0
            r_history(angle_idx) = NaN; % 如果时长无效，则响应值无效
        else
            % 响应值 = 积分值 / (最大可能贡献 * 时长)
            r_history(angle_idx) = integral_value / (v0 * duration);
        end
    end

    % 计算本次运行的总响应性 R (所有有效 r(phi) 的平均值)
    R_values(run_idx) = mean(r_history(~isnan(r_history)));

    % --- 更新单次运行结果的可视化 ---
    % 绘制速度投影曲线
    subplot(ax_proj);
    cla(ax_proj);
    plot(ax_proj, time_vec, projection_history);
    xlabel(ax_proj, '时间 (s)');
    ylabel(ax_proj, 'V(t) · n_φ');
    title(ax_proj, '平均速度在各方向上的投影');
    grid(ax_proj, 'on');

    % 绘制 r(phi) 的条形图
    subplot(ax_resp);
    cla(ax_resp);
    bar(ax_resp, r_history);
    xlabel(ax_resp, '方向样本索引');
    ylabel(ax_resp, 'r(φ)');
    title(ax_resp, sprintf('单次运行响应性 r(φ) (平均 R = %.3f)', R_values(run_idx)));
    grid(ax_resp, 'on');
end

%% 5. 最终统计与可视化 -------------------------------------------------------------
% 所有运行结束后，在右下角子图中绘制 R 值的统计直方图
subplot(ax_stats);
cla(ax_stats);
valid_R = R_values(~isnan(R_values)); % 筛选出所有有效的 R 值

if isempty(valid_R)
    % 如果没有任何有效的 R 值
    text(0.1, 0.5, '无有效响应数据', 'Parent', ax_stats, 'FontSize', 12);
else
    % 绘制直方图
    histogram(ax_stats, valid_R, 'FaceColor', [0.2 0.6 0.8]);
    xlabel(ax_stats, '响应性 R');
    ylabel(ax_stats, '频数');
    title(ax_stats, sprintf('所有运行的 R 值分布 (均值: %.3f)', mean(valid_R)));
    grid(ax_stats, 'on');
end

% 在命令行输出最终统计结果
fprintf('响应性统计完成。有效样本 %d 个，平均 R = %.3f\n', numel(valid_R), mean(valid_R));

%% 辅助函数 ========================================================================
function V = compute_average_velocity(theta, v0)
    % COMPUTE_AVERAGE_VELOCITY - 计算粒子群的平均速度向量
    % 输入:
    %   theta - 所有粒子的方向角向量 (弧度)
    %   v0    - 粒子的标量速度
    % 输出:
    %   V     - 1x2 的平均速度向量 [Vx, Vy]
    
    % 计算所有粒子速度向量的 x 和 y 分量的平均值
    V = [mean(v0 * cos(theta)), mean(v0 * sin(theta))];
end
