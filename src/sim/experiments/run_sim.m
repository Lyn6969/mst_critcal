% run_sim 粒子群运动仿真和可视化脚本
%
% 功能描述:
%   该脚本实现了粒子群运动的基础仿真和可视化功能，展示粒子
%   在二维空间中的集体运动行为和激活传播过程
%
% 主要功能:
%   - 粒子群运动仿真
%   - 实时可视化显示
%   - 阶参数和激活个体数统计
%   - 粒子状态和方向显示
%
% 使用方法:
%   1. 直接运行脚本开始仿真
%   2. 观察左侧粒子运动和右侧统计图表
%   3. 仿真结束后可分析阶参数变化趋势
%
% 输出结果:
%   - 实时动画显示粒子运动
%   - 阶参数随时间变化曲线
%   - 激活个体数随时间变化曲线
%
% 作者：系统生成
% 日期：2025年
% 版本：MATLAB 2025a兼容

clc;        % 清除命令行窗口
clear;      % 清除工作空间变量
close all;  % 关闭所有图形窗口

%% 1. 设置图形默认属性
% 统一设置图形界面的字体、线宽等属性，确保所有图形具有一致的外观

set(0, 'DefaultAxesFontSize', 11);      % 设置坐标轴字体大小
set(0, 'DefaultAxesFontName', 'Arial');  % 设置坐标轴字体类型
set(0, 'DefaultTextFontSize', 11);       % 设置文本字体大小
set(0, 'DefaultTextFontName', 'Arial');  % 设置文本字体类型
set(0, 'DefaultLineLineWidth', 1.5);     % 设置线条默认宽度

%% 2. 参数设置
% 配置粒子仿真系统的各项参数，包括粒子属性、环境参数和仿真控制参数

% 基本粒子参数
params.N = 200;                    % 粒子数量
params.rho = 1;                    % 粒子密度
params.v0 = 1;                     % 粒子速度
params.angleUpdateParameter = 10;    % 角度更新参数
params.angleNoiseIntensity = 0.2;    % 角度噪声强度
params.T_max = 2000;               % 最大仿真时间步数
params.dt = 0.1;                   % 时间步长
params.cj_threshold = 10;          % 激活阈值（弧度/时间）
params.radius = 5;                 % 邻居查找半径
params.deac_threshold = 0.1745;    % 取消激活的角度阈值（弧度，约10度）

% 固定场地参数
params.fieldSize = 100;              % 方形场地边长
params.initDirection = pi/4;        % 45度初始方向
params.useFixedField = true;        % 启用固定场地模式

% 可视化控制
enable_visualization = true;        % 是否启用实时可视化

%% 3. 创建仿真对象
% 使用预设参数创建粒子仿真对象，该类实现了粒子群运动的核心算法

simulation = ParticleSimulation(params);

%% 4. 创建可视化界面
% 如果启用可视化，则创建包含三个子图的图形界面：粒子运动显示、阶参数变化和激活个体数变化

if enable_visualization
    % 创建主图形窗口
    main_figure = figure('Name', '集群运动仿真系统', 'NumberTitle', 'off', ...
        'Position', [100, 100, 1400, 600], ...
        'Color', 'white');

    % 设置子图位置（扩大主显示区域）
    p1 = subplot('Position', [0.05 0.1 0.6 0.8]);     % 左侧图（扩大）
    p2 = subplot('Position', [0.7 0.55 0.25 0.35]);   % 右上图
    p3 = subplot('Position', [0.7 0.1 0.25 0.35]);    % 右下图

    % 存储参数和仿真对象到 figure 的 UserData
    main_figure.UserData.params = params;
    main_figure.UserData.simulation = simulation;
    main_figure.UserData.plots = struct();

    % 左侧：粒子运动和激活状态
    subplot(p1);
    hold on;
    ax1 = gca;
    ax1.Box = 'on';
    ax1.LineWidth = 1.2;
    ax1.GridAlpha = 0.15;
    ax1.GridLineStyle = ':';

    % 智能坐标轴适配
    if params.useFixedField
        axis_range = [0 params.fieldSize 0 params.fieldSize];
        grid_spacing = max(1, params.fieldSize/20);
    else
        axis_range = [0 simulation.simulationAreaSize 0 simulation.simulationAreaSize];
        grid_spacing = max(1, simulation.simulationAreaSize/20);
    end

    axis(axis_range);
    xticks(0:grid_spacing:axis_range(2));
    yticks(0:grid_spacing:axis_range(4));
    xlabel('X Position', 'FontWeight', 'bold');
    ylabel('Y Position', 'FontWeight', 'bold');
    title('Particle Motion & Activation State', 'FontWeight', 'bold', 'FontSize', 12);
    grid on;
    axis square;

    % 创建散点图
    particles_plot = scatter(simulation.positions(:,1), simulation.positions(:,2), 25, ...
        simulation.theta, 'filled', 'MarkerFaceAlpha', 0.7);
    activated_plot_markers = scatter([], [], 60, [0.85 0.1 0.2], 'x', 'LineWidth', 2);

    % 创建箭头
    u = cos(simulation.theta);
    v = sin(simulation.theta);
    arrows_plot = quiver(simulation.positions(:,1), simulation.positions(:,2), ...
        u, v, 0.3, 'Color', [0.5 0.5 0.5 0.6], 'LineWidth', 1, 'MaxHeadSize', 0.8);

    % 设置颜色映射
    colormap(ax1, hsv);
    c = colorbar;
    c.Label.String = 'θ (rad)';
    c.Label.FontWeight = 'bold';
    clim([0 2*pi]);
    c.Ticks = 0:pi/2:2*pi;
    c.TickLabels = {'0°', '90°', '180°', '270°', '360°'};

    % 右上：阶参数变化
    subplot(p2);
    hold on;
    ax2 = gca;
    ax2.Box = 'on';
    ax2.LineWidth = 1.2;
    ax2.GridAlpha = 0.15;
    ax2.GridLineStyle = ':';
    xlabel('Time Steps', 'FontWeight', 'bold');
    ylabel('Order Parameter', 'FontWeight', 'bold');
    title('System Order Evolution', 'FontWeight', 'bold', 'FontSize', 12);
    grid on;
    ylim([0, 1]);
    xlim([0, params.T_max]);
    order_plot = plot(NaN, NaN, 'Color', [0.2 0.4 0.8], 'LineWidth', 1.5);
    hold off;

    % 右下：激活个体数变化
    subplot(p3);
    hold on;
    ax3 = gca;
    ax3.Box = 'on';
    ax3.LineWidth = 1.2;
    ax3.GridAlpha = 0.15;
    ax3.GridLineStyle = ':';
    xlabel('Time Steps', 'FontWeight', 'bold');
    ylabel('Active Particles', 'FontWeight', 'bold');
    title('Active Population Dynamics', 'FontWeight', 'bold', 'FontSize', 12);
    grid on;
    ylim([0, params.N]);
    xlim([0, params.T_max]);
    activated_plot = plot(NaN, NaN, 'Color', [0.85 0.1 0.2], 'LineWidth', 1.5);
    hold off;

    % 存储图形句柄
    main_figure.UserData.plots.particles = particles_plot;
    main_figure.UserData.plots.arrows = arrows_plot;
    main_figure.UserData.plots.activated = activated_plot_markers;
    main_figure.UserData.plots.order = order_plot;
    main_figure.UserData.plots.activated_count = activated_plot;
end

%% 5. 主仿真循环
% 执行完整的仿真过程，包括粒子状态更新和可视化界面刷新

% 开始计时
tic;

% 主循环：遍历每个时间步
for t_step = 1:params.T_max
    % 执行仿真步骤
    simulation.step();

    % 显示进度信息（每100步或最后一步）
    if mod(t_step, 100) == 0 || t_step == params.T_max
        fprintf('Progress: %d/%d steps (%.1f%%)\n', t_step, params.T_max, t_step/params.T_max*100);
    end

    % 更新可视化界面（每2步更新一次，提高性能）
    if enable_visualization && mod(t_step, 2) == 0
        % 获取当前状态
        positions = simulation.positions;
        theta = simulation.theta;
        is_activated = simulation.isActive;

        % 更新粒子位置和颜色
        subplot(p1);
        set(particles_plot, 'XData', positions(:,1), 'YData', positions(:,2), 'CData', theta);

        % 更新箭头
        u = 0.3 * cos(theta);
        v = 0.3 * sin(theta);
        set(arrows_plot, 'XData', positions(:,1), 'YData', positions(:,2), ...
            'UData', u, 'VData', v);

        % 更新激活标记
        activated_positions = positions(is_activated, :);
        set(activated_plot_markers, 'XData', activated_positions(:,1), ...
            'YData', activated_positions(:,2));

        % 更新阶参数图
        subplot(p2);
        set(order_plot, 'XData', 1:t_step, 'YData', simulation.order_parameter(1:t_step));

        % 更新激活个体数图
        subplot(p3);
        set(activated_plot, 'XData', 1:t_step, 'YData', simulation.activated_counts(1:t_step));

        % 刷新图形显示
        drawnow limitrate;
    end
end

% 仿真完成，显示总用时
fprintf('仿真完成！总用时: %.2f 秒\n', toc);

