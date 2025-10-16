% run_external_pulse_test 外源激活机制测试脚本
%
% 功能描述:
%   该脚本专门用于测试和演示外源激活机制，通过可视化界面
%   展示外源脉冲如何触发级联传播过程
%
% 主要功能:
%   - 外源脉冲触发机制演示
%   - 级联传播过程可视化
%   - 外源激活粒子状态跟踪
%   - 系统响应分析
%
% 实验流程:
%   1. 系统稳定期运行
%   2. 触发外源脉冲(强制转向)
%   3. 级联传播过程
%   4. 系统恢复平衡
%
% 输出结果:
%   - 粒子状态实时显示
%   - 外源激活和常规激活对比
%   - 阶参数和激活个体数变化
%   - 外源激活效果分析
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
% 配置粒子仿真系统和外源激活的各项参数

% 基本粒子参数
params.N = 200;                     % 粒子数量（减少以便观察）
params.rho = 1;                     % 粒子密度
params.v0 = 1;                      % 粒子速度
params.angleUpdateParameter = 10;   % 角度更新参数
params.angleNoiseIntensity = 0;   % 角度噪声强度（关闭以便观察级联）
params.T_max = 400;                 % 减少仿真时间以便观察外源激活
params.dt = 0.1;                    % 时间步长
params.cj_threshold = 0.5;           % 激活阈值（较低以便触发级联）
params.radius = 5;                  % 邻居查找半径
params.deac_threshold = 0.1745;     % 取消激活阈值

% 固定场地参数
params.fieldSize = 60;              % 方形场地边长
params.initDirection = pi/4;        % 45度初始方向
params.useFixedField = true;        % 启用固定场地模式

% 外源激活参数
params.stabilization_steps = 200;    % 缩短稳定期以便观察
params.external_pulse_count = 2;    % 激活2个个体
params.forced_turn_duration = 300;   % 独立状态持续时间

% 可视化控制
enable_visualization = true;         % 是否启用实时可视化
%% 3. 创建仿真对象
% 使用预设参数创建带外源激活功能的粒子仿真对象

simulation = ParticleSimulationWithExternalPulse(params);

% 显示实验配置信息
fprintf('=== 外源激活机制测试 ===\n');
fprintf('粒子数量: %d\n', params.N);
fprintf('稳定期: %d 步\n', params.stabilization_steps);
fprintf('外源激活个体数: %d\n', params.external_pulse_count);
fprintf('强制转向后独立时间: %d 步\n', params.forced_turn_duration);

%% 4. 创建可视化界面
% 如果启用可视化，则创建包含三个子图的图形界面：粒子运动显示、阶参数变化和激活个体数变化

if enable_visualization
    % 创建主图形窗口
    main_figure = figure('Name', '外源激活机制测试', 'NumberTitle', 'off', ...
        'Position', [100, 100, 1600, 800], ...
        'Color', 'white');

    % 设置子图位置
    p1 = subplot('Position', [0.05 0.1 0.5 0.8]);      % 左侧主显示区域
    p2 = subplot('Position', [0.6 0.55 0.35 0.35]);    % 右上：阶参数
    p3 = subplot('Position', [0.6 0.1 0.35 0.35]);     % 右下：激活个体数

    % 左侧：粒子运动和激活状态
    subplot(p1);
    hold on;
    ax1 = gca;
    ax1.Box = 'on';
    ax1.LineWidth = 1.2;
    ax1.GridAlpha = 0.15;
    ax1.GridLineStyle = ':';

    % 智能坐标轴适配
    axis_range = [0 params.fieldSize 0 params.fieldSize];
    grid_spacing = max(1, params.fieldSize/20);

    axis(axis_range);
    xticks(0:grid_spacing:axis_range(2));
    yticks(0:grid_spacing:axis_range(4));
    xlabel('X Position', 'FontWeight', 'bold');
    ylabel('Y Position', 'FontWeight', 'bold');
    title('粒子运动 & 外源激活状态', 'FontWeight', 'bold', 'FontSize', 12);
    grid on;
    axis square;

    % 创建散点图
    particles_plot = scatter(simulation.positions(:,1), simulation.positions(:,2), 30, ...
        simulation.theta, 'filled', 'MarkerFaceAlpha', 0.7);

    % 激活个体标记（红色X）
    activated_plot_markers = scatter([], [], 80, [0.85 0.1 0.2], 'x', 'LineWidth', 3);

    % 外源激活个体标记（绿色圆圈）
    external_plot_markers = scatter([], [], 100, [0.1 0.8 0.1], 'o', 'LineWidth', 3);

    % 创建箭头
    u = cos(simulation.theta);
    v = sin(simulation.theta);
    arrows_plot = quiver(simulation.positions(:,1), simulation.positions(:,2), ...
        u, v, 0.4, 'Color', [0.5 0.5 0.5 0.6], 'LineWidth', 1.2, 'MaxHeadSize', 0.8);

    % 设置颜色映射
    colormap(ax1, hsv);
    c = colorbar;
    c.Label.String = '角度 θ (rad)';
    c.Label.FontWeight = 'bold';
    clim([0 2*pi]);
    c.Ticks = 0:pi/2:2*pi;
    c.TickLabels = {'0°', '90°', '180°', '270°', '360°'};

    % 添加图例
    legend_handles = [activated_plot_markers, external_plot_markers];
    legend_labels = {'常规激活', '外源激活'};
    legend(legend_handles, legend_labels, 'Location', 'northeast');

    % 右上：阶参数变化
    subplot(p2);
    hold on;
    ax2 = gca;
    ax2.Box = 'on';
    ax2.LineWidth = 1.2;
    ax2.GridAlpha = 0.15;
    ax2.GridLineStyle = ':';
    xlabel('时间步', 'FontWeight', 'bold');
    ylabel('阶参数', 'FontWeight', 'bold');
    title('系统序参数演化', 'FontWeight', 'bold', 'FontSize', 12);
    grid on;
    ylim([0, 1]);
    xlim([0, params.T_max]);
    order_plot = plot(NaN, NaN, 'Color', [0.2 0.4 0.8], 'LineWidth', 1.8);

    % 添加外源激活触发线
    xline(params.stabilization_steps + 1, '--r', '外源脉冲', 'LineWidth', 2);
    hold off;

    % 右下：激活个体数变化
    subplot(p3);
    hold on;
    ax3 = gca;
    ax3.Box = 'on';
    ax3.LineWidth = 1.2;
    ax3.GridAlpha = 0.15;
    ax3.GridLineStyle = ':';
    xlabel('时间步', 'FontWeight', 'bold');
    ylabel('激活个体数', 'FontWeight', 'bold');
    title('激活个体动态', 'FontWeight', 'bold', 'FontSize', 12);
    grid on;
    ylim([0, params.N]);
    xlim([0, params.T_max]);
    activated_plot = plot(NaN, NaN, 'Color', [0.85 0.1 0.2], 'LineWidth', 1.8, 'DisplayName', '常规激活');
    external_activated_plot = plot(NaN, NaN, 'Color', [0.1 0.8 0.1], 'LineWidth', 1.8, 'DisplayName', '外源激活');

    % 添加外源激活触发线
    xline(params.stabilization_steps + 1, '--r', '外源脉冲', 'LineWidth', 2);
    legend('show', 'Location', 'northeast');
    hold off;
end

%% 5. 主仿真循环
% 执行完整的仿真过程，包括稳定期、外源脉冲触发和级联传播

% 开始计时
tic;

% 预分配外源激活个体数存储数组
external_counts = zeros(params.T_max, 1);

% 主循环：遍历每个时间步
for t_step = 1:params.T_max
    % 执行仿真步骤
    simulation.step();

    % 记录外源激活个体数
    external_counts(t_step) = simulation.getExternallyActivatedCount();

    % 显示关键步骤进度
    if t_step == params.stabilization_steps
        fprintf('步骤 %d: 稳定期结束\n', t_step);
    elseif t_step == params.stabilization_steps + 1
        fprintf('步骤 %d: 外源脉冲触发\n', t_step);
    elseif mod(t_step, 50) == 0 || t_step == params.T_max
        fprintf('进度: %d/%d 步 (%.1f%%), 外源激活个体: %d\n', ...
            t_step, params.T_max, t_step/params.T_max*100, external_counts(t_step));
    end

    % 更新可视化界面（每2步更新一次，提高性能）
    if enable_visualization && mod(t_step, 2) == 0
        % 获取当前状态
        positions = simulation.positions;
        theta = simulation.theta;
        is_activated = simulation.isActive;
        external_indices = simulation.getExternallyActivatedIndices();

        % 更新粒子位置和颜色
        subplot(p1);
        set(particles_plot, 'XData', positions(:,1), 'YData', positions(:,2), 'CData', theta);

        % 更新箭头
        u = 0.4 * cos(theta);
        v = 0.4 * sin(theta);
        set(arrows_plot, 'XData', positions(:,1), 'YData', positions(:,2), ...
            'UData', u, 'VData', v);

        % 更新常规激活标记
        activated_positions = positions(is_activated, :);
        set(activated_plot_markers, 'XData', activated_positions(:,1), ...
            'YData', activated_positions(:,2));

        % 更新外源激活标记
        if ~isempty(external_indices)
            external_positions = positions(external_indices, :);
            set(external_plot_markers, 'XData', external_positions(:,1), ...
                'YData', external_positions(:,2));
        else
            set(external_plot_markers, 'XData', [], 'YData', []);
        end

        % 更新阶参数图
        subplot(p2);
        set(order_plot, 'XData', 1:t_step, 'YData', simulation.order_parameter(1:t_step));

        % 更新激活个体数图
        subplot(p3);
        set(activated_plot, 'XData', 1:t_step, 'YData', simulation.activated_counts(1:t_step));
        set(external_activated_plot, 'XData', 1:t_step, 'YData', external_counts(1:t_step));

        % 刷新图形显示
        drawnow limitrate;
    end
end

% 计算总仿真时间
simulation_time = toc;
fprintf('\n=== 仿真完成 ===\n');
fprintf('总用时: %.2f 秒\n', simulation_time);
fprintf('最终激活个体数: %d\n', sum(simulation.isActive));
fprintf('最终外源激活个体数: %d\n', simulation.getExternallyActivatedCount());

%% 6. 结果分析
% 分析外源激活的效果，比较脉冲前后的系统状态变化

pulse_step = params.stabilization_steps + 1;
if pulse_step <= params.T_max
    % 计算脉冲前后的平均阶参数
    pre_pulse_order = mean(simulation.order_parameter(max(1, pulse_step-10):pulse_step-1));
    post_pulse_order = mean(simulation.order_parameter(pulse_step:min(params.T_max, pulse_step+20)));

    % 显示分析结果
    fprintf('\n=== 结果分析 ===\n');
    fprintf('外源脉冲前阶参数: %.4f\n', pre_pulse_order);
    fprintf('外源脉冲后阶参数: %.4f\n', post_pulse_order);
    fprintf('阶参数变化: %.4f\n', post_pulse_order - pre_pulse_order);

    % 判断外源激活效果
    if post_pulse_order > pre_pulse_order + 0.1
        fprintf('✓ 外源激活成功触发级联效应\n');
    else
        fprintf('⚠ 外源激活效果不明显，可能需要调整参数\n');
    end
end