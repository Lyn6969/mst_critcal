%% 初始化
clear;
clc;
close all;
font_size = 22;
font_weight = "bold";
% 设置图形默认属性
set(0, 'DefaultAxesFontSize',font_size);
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontSize', font_size);
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultLineLineWidth', 1.5);


% 创建保存目录
formats = {'pdf','eps','png'};
for fmt = formats
    if ~exist(['figs/snap/', fmt{1}], 'dir')
        mkdir(['figs/snap/', fmt{1}]);
    end
end

for fmt = formats
    if ~exist(['figs/op_evo/', fmt{1}], 'dir')
        mkdir(['figs/op_evo/', fmt{1}]);
    end
    if ~exist(['figs/fa_evo/', fmt{1}], 'dir')
        mkdir(['figs/fa_evo/', fmt{1}]);
    end
end
%% 参数设置
params = struct();
params.N = 200;
params.rho = 1;
params.v0 = 1;
params.angleUpdateParameter = 3;
eta = 1.3;  % 控制eta
params.T_max = 2000;
params.dt = 0.1;
params.radius = 5;
params.deac_threshold = 0.1745;
params.angleNoiseIntensity = (eta^2)/2;  % 从eta计算DTheta

% 定义不同的MT值和时间点
MT_values = [1,3,8];  % MT值
snapshot_times = [500,1000,2000];  % 快照时间点

% 绘图参数
head_radius = 0.2;          % 头部半径
tail_length = 0.4;          % 尾巴长度
tail_width = 0.1;           % 尾巴宽度
semicircle_resolution = 100; % 半圆的分辨率
ring_thickness = 0.02;
ring_color = 'w';
% 创建颜色映身

cmap = hsv(1440);
num_colors = size(cmap, 1);
%% 绘制snapshot图

mt_idx = 3;
params.cj_threshold = MT_values(mt_idx);
sim = ParticleSimulation(params);

% 创建图形
fig = figure('Position', [100, 100, 1000, 400]);

% 创建自定义colormap
custom_colormap = hsv(1440);

% 首先创建所有子图并设置合适的位置
for t_idx = 1:length(snapshot_times)
    ax = subplot(1, length(snapshot_times), t_idx);
    pos = ax.Position;
    % 调整子图位置，确保有足够空间显示colorbar
    % ax.Position = [pos(1), 0.15, pos(3), 0.65];

    % 设置边框粗细
    ax.LineWidth = 2;

    % 只在第一个子图显示y轴标签
    if t_idx == 1
        ylabel('y', 'FontWeight', font_weight, 'FontSize', font_size);
    else
        ax.YTickLabel = [];  % 移除其他子图的y轴标签
    end

    % 设置x轴标签
    xlabel('x', 'FontWeight', font_weight, 'FontSize', font_size);

    % 设置刻度标签的字体大小
    ax.XAxis.FontSize = font_size;
    ax.YAxis.FontSize = font_size;
    ax.XAxis.FontWeight = font_weight;
    ax.YAxis.FontWeight = font_weight;
end

% 获取最左边和最右边子图的位置
left_subplot = subplot(1, length(snapshot_times), 1);
right_subplot = subplot(1, length(snapshot_times), length(snapshot_times));
left_pos = left_subplot.Position;
right_pos = right_subplot.Position;

% 创建更宽的colorbar并确保其可见
if mt_idx == 1
    colorbar_ax = axes('Position', [left_pos(1), 0.95, ...
        (right_pos(1) + right_pos(3) - left_pos(1)), 0.04]);
    % 设置颜色映射范围
    clim([-pi, pi]);

    % 设置线性色谱，如 parula 或 jet
    colormap(colorbar_ax, custom_colormap);
    c = colorbar(colorbar_ax, 'horizontal');

    % 获取 colorbar 位置
    c.Position(4) = c.Position(4) * 2; % 将 colorbar 的高度增加一倍
    c.Ticks = [-pi,0, pi];
    c.TickLabels = {'180°','0°','-180°'};
    c.LineWidth = 1.5;  % 加粗colorbar边框
    c.FontWeight = font_weight;  % 加粗colorbar刻度标签
    c.FontSize = 15;  % 设置colorbar刻度标签字体大小


    % c.Label.String = 'Angle (deg)'; % 设置标题文本
    % c.Label.FontSize = 20; % 设置字体大小
    % c.Label.FontWeight = 'bold'; % 设置字体粗细
    % c.Label.FontName = 'Times New Roman'; % 设置字体名称
    % c.Label.Position(2) = c.Label.Position(2) + 0.5; % 将标题向上移动

    % 隐藏 colorbar 轴的刻度和标签
    set(colorbar_ax, 'Visible', 'off');
end

% 运行模拟
for t = 1:params.T_max
    sim.step();

    if ismember(t, snapshot_times)
        subplot_idx = find(snapshot_times == t);
        ax = subplot(1, length(snapshot_times), subplot_idx);
        pos = ax.Position;
        ax.Position = [pos(1), 0.1, pos(3), 0.7];
        hold on;

        % 获取数据
        positions = sim.positions;
        theta = sim.theta;

        % 计算序参量和其他参数
        order_param = sim.order_parameter(t);
        active_frac = sum(sim.isActive)/params.N;


        for i = 1:sim.N
            % 获取粒子状态
            pos = sim.positions(i, :);
            theta = -sim.theta(i);

            % 计算颜色（基于角度）
            theta_deg = rad2deg(theta);
            if theta_deg > 180
                theta_deg = theta_deg - 360;
            elseif theta_deg <= -180
                theta_deg = theta_deg + 360;
            end
            color_idx = mod(round((theta_deg + 180) / 360 * num_colors), num_colors) + 1;
            particle_color = cmap(color_idx, :);
            % 绘制尾巴（矩形）
            tail_corners = [
                0, -tail_width/2;
                tail_length, -tail_width/2;
                tail_length, tail_width/2;
                0, tail_width/2
                ];

            % 旋转矩阵
            R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
            rotated_tail = ( R * tail_corners' )';
            rotated_tail = rotated_tail + pos;
            alpha_val = 1;  % 激活个体完全不透明

            % 绘制尾巴
            patch(rotated_tail(:,1), rotated_tail(:,2), particle_color, ...
                'EdgeColor', particle_color, ...
                'FaceAlpha', alpha_val, ...
                'EdgeAlpha', alpha_val);  % 添加边缘透明度

            % 绘制头部（实心圆）
            rectangle('Position', [pos(1)-head_radius, pos(2)-head_radius, 2*head_radius, 2*head_radius], ...
                'Curvature', [1, 1], ...
                'FaceColor', particle_color, ...
                'EdgeColor', particle_color, ...
                'FaceAlpha', alpha_val);  % 添加边缘透明度

            if sim.isActive(i)
                % 在头部中心添加一个小点
                inner_dot_radius = head_radius * 0.5;
                tv = 0:pi/20:2*pi;
                patch(inner_dot_radius*cos(tv) + pos(1), inner_dot_radius*sin(tv) + pos(2), 'k', ...
                    'EdgeColor', 'none');
            end


            % 绘制白色圆环
            rectangle('Position', [pos(1)-head_radius-ring_thickness, pos(2)-head_radius-ring_thickness, ...
                2*(head_radius + ring_thickness), 2*(head_radius + ring_thickness)], ...
                'Curvature', [1, 1], 'EdgeColor', ring_color, 'LineWidth', ring_thickness, 'FaceColor', 'none');

            % 绘制尾巴末端的半圆
            semicircle_radius = tail_width / 2;
            semicircle_angles = linspace(-90, 90, semicircle_resolution);
            semicircle_x = semicircle_radius * cosd(semicircle_angles);
            semicircle_y = semicircle_radius * sind(semicircle_angles);
            semicircle_points = [semicircle_x; semicircle_y]';

            R_semicircle = [cos(theta), -sin(theta); sin(theta), cos(theta)];
            rotated_semicircle = (R_semicircle * semicircle_points')';

            tail_end_center = (rotated_tail(2, :) + rotated_tail(3, :)) / 2;

            rotated_semicircle(:,1) = rotated_semicircle(:,1) + tail_end_center(1);
            rotated_semicircle(:,2) = rotated_semicircle(:,2) + tail_end_center(2);

            patch(rotated_semicircle(:,1), rotated_semicircle(:,2), particle_color, ...
                'EdgeColor', particle_color, ...
                'FaceAlpha', alpha_val, ...
                'EdgeAlpha', alpha_val);  %
        end

        % 设置图形属性
        box on;
        axis square;
        axis([0 sqrt(params.N/params.rho) 0 sqrt(params.N/params.rho)]);
        title(sprintf('t = %d  φ = %.2f  f_{a} = %.2f', ...
            t, order_param, active_frac), ...
            'FontSize', 15, ...
            'FontWeight', 'bold', ...
            'FontName', 'Times New Roman');
        clim(ax, [-pi pi]);
    end
end

% 保存图片
for fmt = formats
    % 修改文件名格式，添加N的信息
    filename = sprintf('figs/snap/%s/snapshots_N%d_MT%.1f.%s', fmt{1}, params.N, MT_values(mt_idx), fmt{1});
    switch fmt{1}
        case 'png'
            % 导出为高分辨率 PNG 位图，设置透明背景
            exportgraphics(fig, filename, 'Resolution', 800);
        case 'pdf'
            % 导出为 PDF 矢量图，设置透明背景
            exportgraphics(fig, filename, 'ContentType', 'vector');
        case 'eps'
            % 导出为 EPS 矢量图，设置透明背景
            exportgraphics(fig, filename, 'ContentType', 'vector');
        otherwise
            warning('Unsupported format: %s', fmt{1});
    end
end
close(fig);

%% 绘制序参量的演化图
% 时间步数
time_points = 1:params.T_max;
% 为序参量φ图设置不同的红色
red_colors = [
    1.0 0.6 0.6   % 浅红
    1.0 0.4 0.4;  % 中红
    0.8 0.2 0.2;  % 深红
    ];

% 绘制序参量φ的演化图
fig_phi = figure('Position', [100, 100, 800, 350]);  % 调整宽度以匹配snapshot图
plot(time_points, sim.order_parameter, 'LineWidth', 2, 'Color', [0.2 0.4 0.8],'HandleVisibility', 'off');

hold on;
% 标记snapshot时间点
for i = 1:length(snapshot_times)
    t = snapshot_times(i);
    % 创建带标签的标记点，使用不同的红色
    plot(t, sim.order_parameter(t), 'o', 'MarkerSize', 10, ...
        'MarkerEdgeColor', red_colors(i,:), ...
        'MarkerFaceColor', 'none', 'LineWidth', 2, ...
        'DisplayName', sprintf('t = %d', t));

    % 画虚线
    % plot([t t], [0 sim.order_parameter(t)], '--', ...
    %     'Color', red_colors(i,:), 'LineWidth', 1, ...
    %     'HandleVisibility', 'off');
end

% 添加图例
legend('Location', 'best','FontSize', font_size, 'FontWeight', font_weight, ...
    'FontName', 'Times New Roman', 'Box', 'on');
ax = gca;
ax.LineWidth = 1.5;
ax.XAxis.FontWeight = font_weight;
ax.YAxis.FontWeight = font_weight;
ax.FontSize = font_size;
xlabel('Time Step', 'FontWeight', font_weight, 'FontSize', font_size);
ylabel('φ', 'FontSize', font_size, 'Interpreter', 'tex','FontName','Arial');
% grid off;
box on;
ylim([0 1]);
% axis equal;
if mt_idx == 1
    title('Time Evolution of Order Parameter', ...
        'FontWeight', font_weight, 'FontSize', font_size);
end
% 保存序参量演化图
for fmt = formats
    filename = sprintf('figs/op_evo/%s/op_evolution_N%d_MT%.1f.%s', ...
        fmt{1}, params.N, MT_values(mt_idx), fmt{1});
    exportgraphics(fig_phi, filename, 'Resolution', 800);
end
% close(fig_phi);

%% 绘制活跃分数fa的演化图
% 同样为fa图添加标记
% 为活跃分数fa图设置不同的蓝色
blue_colors = [
    0.6 0.6 1.0   % 浅蓝
    0.4 0.4 1.0;  % 中蓝
    0.2 0.2 0.8;  % 深蓝
    ];
fig_fa = figure('Position', [100, 100, 800, 350]);
plot(time_points, sim.activated_counts/params.N, 'LineWidth', 2, 'Color', [0.8 0.3 0.3], 'HandleVisibility','off');
hold on;
for i = 1:length(snapshot_times)
    t = snapshot_times(i);
    % 创建带标签的标记点，使用不同的蓝色
    plot(t, sim.activated_counts(t)/params.N, 'o', 'MarkerSize', 10, ...
        'MarkerEdgeColor', blue_colors(i,:), ...
        'MarkerFaceColor', 'none', 'LineWidth', 2, ...
        'DisplayName', sprintf('t = %d', t));

    % 画虚线
    % plot([t t], [0 sim.activated_counts(t)/params.N], '--', ...
    %     'Color', blue_colors(i,:), 'LineWidth', 1, ...
    %     'HandleVisibility', 'off');
end

% 两个图都添加图例
legend('Location', 'best', 'FontSize', font_size, 'FontWeight', font_weight, ...
    'FontName', 'Times New Roman', 'Box', 'on');
ax = gca;
ax.LineWidth = 1.5;
ax.XAxis.FontWeight = font_weight;
ax.YAxis.FontWeight = font_weight;
ax.FontSize = font_size;

xlabel('Time Step', 'FontWeight', font_weight, 'FontSize', font_size);
ylabel('f_{a}', 'FontWeight', font_weight, 'FontSize', font_size);
box on;
ylim([0 1]);
if mt_idx ==1
    title('Time Evolution of Active Fraction', ...
        'FontWeight', font_weight, 'FontSize', font_size);
end
% 保存活跃分数演化图
for fmt = formats
    filename = sprintf('figs/fa_evo/%s/fa_evolution_N%d_MT%.1f.%s', ...
        fmt{1}, params.N, MT_values(mt_idx), fmt{1});
    exportgraphics(fig_fa, filename, 'Resolution', 800);
end