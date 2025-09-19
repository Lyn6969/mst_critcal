function [final_positions, final_velocities] = randPose(num_agents)
    % 生成不重叠的个体位置和速度 (高效向量化版本)
    % 通过物理仿真让初始随机位置的个体自然分散，避免位置重叠
    % 
    % 输入参数:
    %   num_agents - 个体数量
    % 
    % 输出参数:
    %   final_positions - 最终位置矩阵 [2 x num_agents]
    %   final_velocities - 最终速度矩阵 [2 x num_agents]
    
    % === 仿真参数设置 ===
    max_simulation_steps = 10;      % 最大仿真步数
    time_step = 0.1;                % 仿真时间步长 (秒)
    
    % === 力学参数设置 ===
    repulsion_range = 3;            % 排斥力作用范围
    attraction_decay = 10;          % 吸引力衰减系数
    max_acceleration = 5;           % 最大加速度限制
    target_speed = 0;               % 目标速度 (接近静止)
    speed_relaxation_time = 0.1;    % 速度松弛时间
    noise_strength = 0.5;           % 噪声强度
    
    % === 初始化个体状态 ===
    current_positions = rand(2, num_agents) * 3;  % 初始位置随机分布在 [0,2] x [0,2] 区域
    current_velocities = ones(2, num_agents);      % 初始速度接近静止

    % === 物理仿真循环 ===
    for step = 1:max_simulation_steps
        
        % --- 计算自驱力 (速度调节力) - 完全向量化 ---
        current_speeds = vecnorm(current_velocities, 2, 1);  % 使用内置vecnorm函数
        speed_error = (target_speed - current_speeds) / speed_relaxation_time;
        
        % 避免除零的向量化处理
        safe_speeds = max(current_speeds, eps);  % 防止除零
        velocity_directions = current_velocities ./ safe_speeds;
        self_propulsion_force = velocity_directions .* speed_error;
        
        % --- 计算个体间相互作用力 - 使用pdist2优化 ---
        % 使用pdist2计算距离矩阵 (显著更快)
        distance_matrix = pdist2(current_positions', current_positions');
        
        % 避免除零和自身作用
        distance_matrix(distance_matrix == 0) = inf;
        
        % 计算力的大小矩阵
        force_magnitude_matrix = (1 - (repulsion_range ./ distance_matrix).^2) .* ...
                                exp(-distance_matrix / attraction_decay);
        
        % 计算单位方向向量
        pos_x = current_positions(1,:);
        pos_y = current_positions(2,:);
        dx = pos_x - pos_x';  % N x N 矩阵
        dy = pos_y - pos_y';  % N x N 矩阵
        dx_norm = dx ./ distance_matrix;
        dy_norm = dy ./ distance_matrix;
        
        % 计算每个个体受到的总力
        fx_total = sum(force_magnitude_matrix .* dx_norm, 2, 'omitnan')';
        fy_total = sum(force_magnitude_matrix .* dy_norm, 2, 'omitnan')';
        
        interaction_force = [fx_total; fy_total];
        
        % --- 状态更新 - 向量化优化 ---
        total_force = self_propulsion_force + interaction_force;
        
        % 限制加速度大小 - 向量化
        force_magnitudes = vecnorm(total_force, 2, 1);
        scale_factors = min(max_acceleration ./ max(force_magnitudes, eps), 1);
        limited_force = total_force .* scale_factors;
        
        % 添加随机噪声
        noise = (rand(2, num_agents) - 0.5) * 2 * noise_strength;  % 更高效的噪声生成
        acceleration = limited_force + noise;
        
        % 更新速度和位置
        current_velocities = current_velocities + acceleration * time_step;
        current_positions = current_positions + current_velocities * time_step;
    end
    
    % === 返回最终结果 ===
    final_positions = current_positions;
    final_velocities = current_velocities;
end



