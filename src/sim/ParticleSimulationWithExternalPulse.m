classdef ParticleSimulationWithExternalPulse < ParticleSimulation
    properties
        % 外源激活控制参数
        stabilization_steps = 100;           % 稳定运行步数
        external_pulse_count = 1;            % 外源激活个体数量(1或2)
        forced_turn_duration = 50;           % 强制转向后独立状态持续时间
        turn_completion_threshold = 5*pi/180; % 转向完成判断阈值（5度转弧度）

        % 外源激活状态变量
        isExternallyActivated;               % [N x 1] 逻辑数组，标记外源激活个体
        external_target_theta;               % [N x 1] 外源目标角度
        external_activation_timer;           % [N x 1] 剩余独立状态时间
        external_pulse_triggered = false;    % 标记是否已经触发过外源脉冲

        % 级联统计跟踪变量
        everActivated;                       % [N x 1] 布尔数组，记录整个级联过程中曾被激活的个体（永不重置）
        cascade_active = false;              % 布尔值，标记级联是否仍在进行
        steps_since_last_activation = 0;     % 计数器，用于检测级联结束
        cascade_end_threshold = 5;           % 连续N步无新激活则认为级联结束
        previous_activated_count = 0;        % 上一步的激活个体总数，用于检测新激活
    end

    methods
        function obj = ParticleSimulationWithExternalPulse(params)
            % 调用父类构造函数
            obj@ParticleSimulation(params);

            % 初始化外源激活相关状态
            obj.isExternallyActivated = false(obj.N, 1);
            obj.external_target_theta = zeros(obj.N, 1);
            obj.external_activation_timer = zeros(obj.N, 1);

            % 初始化级联统计跟踪变量
            obj.everActivated = false(obj.N, 1);
        end

        function triggerExternalPulse(obj)
            % 触发外源脉冲，随机选择个体并设置90度转向目标
            if obj.external_pulse_triggered
                return; % 已经触发过，避免重复触发
            end

            % 初始化级联统计跟踪
            obj.everActivated = false(obj.N, 1);
            obj.cascade_active = true;
            obj.steps_since_last_activation = 0;
            obj.previous_activated_count = sum(obj.isActive);

            % 随机选择个体进行外源激活
            available_indices = 1:obj.N;
            selected_indices = randsample(available_indices, obj.external_pulse_count);

            for i = 1:length(selected_indices)
                idx = selected_indices(i);

                % 设置外源激活状态
                obj.isExternallyActivated(idx) = true;
                % 将外源激活个体也标记到级联跟踪中
                obj.everActivated(idx) = true;

                % 设置目标角度：当前角度 + 90度
                obj.external_target_theta(idx) = mod(obj.theta(idx) + pi/2, 2*pi);

                % 初始化为强制转向状态（计时器为0表示正在转向）
                obj.external_activation_timer(idx) = 0;

                fprintf('外源激活个体 %d: 当前角度 %.2f° → 目标角度 %.2f°\n', ...
                    idx, rad2deg(obj.theta(idx)), rad2deg(obj.external_target_theta(idx)));
            end

            obj.external_pulse_triggered = true;
            fprintf('步骤 %d: 外源脉冲触发，激活 %d 个个体，级联跟踪开始\n', ...
                obj.current_step, obj.external_pulse_count);
        end

        function desired_theta = updateExternallyActivatedParticles(obj, desired_theta)
            % 更新外源激活个体的状态和期望方向
            % 输入：desired_theta - 原始期望方向数组
            % 输出：desired_theta - 修改后的期望方向数组

            for i = 1:obj.N
                if ~obj.isExternallyActivated(i)
                    continue;
                end

                if obj.external_activation_timer(i) == 0
                    % 正在强制转向阶段
                    % 覆盖期望角度为目标角度，实现强制转向
                    desired_theta(i) = obj.external_target_theta(i);

                    % 添加调试输出
                    current_angle = rad2deg(obj.theta(i));
                    target_angle = rad2deg(obj.external_target_theta(i));
                    fprintf('步骤 %d: 个体 %d 强制转向 - 当前: %.1f° → 目标: %.1f°\n', ...
                        obj.current_step, i, current_angle, target_angle);

                    % 检查是否已经转向完成
                    angle_diff = abs(wrapToPi(obj.theta(i) - obj.external_target_theta(i)));
                    if angle_diff < obj.turn_completion_threshold
                        % 转向完成，进入独立状态
                        obj.external_activation_timer(i) = obj.forced_turn_duration;
                        fprintf('个体 %d 转向完成，进入 %d 步独立状态\n', ...
                            i, obj.forced_turn_duration);
                    end

                elseif obj.external_activation_timer(i) > 0
                    % 独立状态阶段 - 不受邻居影响，保持当前方向
                    desired_theta(i) = obj.theta(i);

                    % 递减计时器
                    obj.external_activation_timer(i) = obj.external_activation_timer(i) - 1;

                    if obj.external_activation_timer(i) == 0
                        % 独立状态结束，返回正常状态
                        obj.isExternallyActivated(i) = false;
                        fprintf('个体 %d 结束外源激活状态，返回正常\n', i);
                    end
                end
            end
        end

        function step(obj)
            % 重写step方法，集成外源激活机制
            if obj.current_step >= obj.T_max
                return;
            end
            obj.current_step = obj.current_step + 1;
            t_step = obj.current_step;

            % === 外源脉冲触发检查 ===
            if obj.current_step == obj.stabilization_steps + 1
                obj.triggerExternalPulse();
            end

            % === 原有的邻居查找和状态更新逻辑 ===
            neighbor_matrix = obj.findNeighbors();
            desired_theta = NaN(obj.N, 1);

            % 2. 更新粒子状态和期望方向（原有逻辑）
            for i = 1:obj.N
                neibor_idx = find(neighbor_matrix(i, :));
                if obj.isActive(i)
                    % 已激活，更新源头 ID
                    if ismember(obj.src_ids{i}, neibor_idx)  % 检查唯一的源头是否还在邻居中
                        % 计算与源头的角度差（弧度）
                        src_direction = obj.theta(obj.src_ids{i});
                        angle_diff_rad = abs(wrapToPi(obj.theta(i) - src_direction));

                        if angle_diff_rad < obj.deac_threshold
                            % 取消激活
                            obj.isActive(i) = false;
                            obj.src_ids{i} = [];
                            if ~isempty(neibor_idx)
                                neibor_directions = obj.theta(neibor_idx);
                                avg_neibor_dir = angle(mean(exp(1j * neibor_directions)));
                                avg_neibor_dir = wrapTo2Pi(avg_neibor_dir);
                                desired_theta(i) = avg_neibor_dir;
                            else
                                desired_theta(i) = obj.theta(i);
                            end
                        else
                            % 保持激活，跟随源头
                            desired_theta(i) = src_direction;
                        end
                    else
                        % 源头不在邻居中，取消激活
                        obj.isActive(i) = false;
                        obj.src_ids{i} = [];
                        if ~isempty(neibor_idx)
                            neibor_directions = obj.theta(neibor_idx);
                            avg_neibor_dir = angle(mean(exp(1j * neibor_directions)));
                            avg_neibor_dir = wrapTo2Pi(avg_neibor_dir);
                            desired_theta(i) = avg_neibor_dir;
                        else
                            desired_theta(i) = obj.theta(i);
                        end
                    end
                else
                    % 未激活，检查候选邻居
                    if isempty(neibor_idx)
                        desired_theta(i) = obj.theta(i);
                    else
                        % 计算相对位置变化
                        current_diff = obj.positions(neibor_idx, :) - obj.positions(i, :);
                        past_diff = obj.previousPositions(neibor_idx, :) - obj.previousPositions(i, :);

                        % 使用标准欧几里得距离计算，不考虑周期性边界条件
                        current_dist = vecnorm(current_diff, 2, 2);
                        past_dist = vecnorm(past_diff, 2, 2);

                        % 计算单位方向向量
                        current_diff_unit = zeros(size(current_diff));
                        past_diff_unit = zeros(size(past_diff));
                        non_zero_current = current_dist > 0;
                        non_zero_past = past_dist > 0;

                        current_diff_unit(non_zero_current, :) = current_diff(non_zero_current, :) ./ current_dist(non_zero_current);
                        past_diff_unit(non_zero_past, :) = past_diff(non_zero_past, :) ./ past_dist(non_zero_past);

                        % 计算角度差（弧度）
                        angle_cos = sum(past_diff_unit .* current_diff_unit, 2);
                        angle_cos = max(min(angle_cos, 1), -1);
                        angle_diff_rad = acos(angle_cos);

                        % 计算显著性值 s
                        s_values = angle_diff_rad / obj.dt;
                        % 找出显著性最高的邻居
                        [max_s, max_s_idx] = max(s_values);

                        if max_s > obj.cj_threshold
                            % 激活并只跟随最显著的邻居
                            obj.isActive(i) = true;
                            obj.src_ids{i} = neibor_idx(max_s_idx);  % 只存储显著性最高的邻居ID
                            desired_theta(i) = obj.theta(neibor_idx(max_s_idx));  % 直接使用该邻居的方向
                        else
                            % 不激活，设置期望方向为邻居平均方向
                            neighbor_directions = obj.theta(neibor_idx);
                            avg_dir = angle(mean(exp(1j * neighbor_directions)));
                            avg_dir = wrapTo2Pi(avg_dir);
                            desired_theta(i) = avg_dir;
                        end
                    end
                end
            end

            % === 外源激活处理（关键新增） ===
            desired_theta = obj.updateExternallyActivatedParticles(desired_theta);

            % === 原有的角度和位置更新逻辑 ===
            % 3. 更新角度
            delta_theta = desired_theta - obj.theta;
            weighted_sin = sin(delta_theta);
            obj.theta = obj.theta + obj.angleUpdateParameter * weighted_sin * obj.dt + sqrt(2 * obj.angleNoiseIntensity * obj.dt) * randn(obj.N, 1);
            obj.theta = mod(obj.theta, 2 * pi);

            % 4. 更新位置
            obj.previousPositions = obj.positions;
            obj.positions = obj.positions + obj.v0 * [cos(obj.theta), sin(obj.theta)] * obj.dt;
            % 移除周期性边界条件：粒子可以移动到边界之外

            % 5. 计算并存储阶参数和激活个体数
            obj.order_parameter(t_step) = norm(sum(exp(1j * obj.theta))) / obj.N;
            obj.activated_counts(t_step) = sum(obj.isActive);

            % === 级联统计更新逻辑 ===
            if obj.cascade_active
                % 更新everActivated数组：记录曾经被激活过的个体
                obj.everActivated = obj.everActivated | obj.isActive;

                % 检测级联是否结束
                current_activated_count = sum(obj.isActive);
                if current_activated_count > obj.previous_activated_count
                    % 有新的激活个体，重置计数器
                    obj.steps_since_last_activation = 0;
                    obj.previous_activated_count = current_activated_count;
                else
                    % 没有新的激活个体，增加计数器
                    obj.steps_since_last_activation = obj.steps_since_last_activation + 1;
                end

                % 判断级联是否结束
                if obj.steps_since_last_activation >= obj.cascade_end_threshold
                    obj.cascade_active = false;
                    cascade_size = sum(obj.everActivated) / obj.N;
                    fprintf('步骤 %d: 级联结束，最终级联规模 = %.4f (%d/%d)\n', ...
                        obj.current_step, cascade_size, sum(obj.everActivated), obj.N);
                end
            end
        end

        function cascade_size = getCascadeSize(obj)
            % 返回当前级联规模（everActivated个体数/总个体数）
            cascade_size = sum(obj.everActivated) / obj.N;
        end

        function is_complete = isCascadeComplete(obj)
            % 判断级联是否完全结束
            is_complete = ~obj.cascade_active;
        end

        function resetCascadeTracking(obj)
            % 重置级联统计跟踪变量，准备下次实验
            obj.everActivated = false(obj.N, 1);
            obj.cascade_active = false;
            obj.steps_since_last_activation = 0;
            obj.previous_activated_count = 0;
            obj.external_pulse_triggered = false;

            % 重置外源激活相关状态
            obj.isExternallyActivated = false(obj.N, 1);
            obj.external_target_theta = zeros(obj.N, 1);
            obj.external_activation_timer = zeros(obj.N, 1);

            % 重置粒子激活状态
            obj.isActive = false(obj.N, 1);
            obj.src_ids = cell(obj.N, 1);
            for i = 1:obj.N
                obj.src_ids{i} = [];
            end

            fprintf('级联统计跟踪已重置\n');
        end

        function cascade_size = runSingleExperiment(obj, initial_count)
            % 运行单次级联实验，支持c1/c2计算
            % 输入：initial_count - 初发个体数量（1或2）
            % 输出：cascade_size - 最终级联规模（everActivated个体数/总个体数）

            if nargin < 2
                initial_count = 1;  % 默认为c1实验（1个初发个体）
            end

            % 重置系统状态
            obj.resetCascadeTracking();

            % 重新初始化粒子位置和方向
            obj.initializeParticles();

            % 设置实验参数
            original_pulse_count = obj.external_pulse_count;
            obj.external_pulse_count = initial_count;

            % 重置时间步
            obj.current_step = 0;

            % 运行稳定期
            fprintf('开始c%d实验：%d个初发个体\n', initial_count, initial_count);
            fprintf('运行稳定期（%d步）...\n', obj.stabilization_steps);

            for step = 1:obj.stabilization_steps
                obj.step();
            end

            % 触发外源脉冲并继续运行直到级联结束
            fprintf('稳定期结束，触发外源脉冲...\n');
            max_cascade_steps = 200;  % 最大级联步数，防止无限循环

            for step = 1:max_cascade_steps
                obj.step();

                % 检查级联是否结束
                if obj.isCascadeComplete()
                    break;
                end
            end

            % 获取最终级联规模
            cascade_size = obj.getCascadeSize();

            % 恢复原始参数
            obj.external_pulse_count = original_pulse_count;

            fprintf('c%d实验完成：级联规模 = %.4f (%d/%d)，总步数 = %d\n', ...
                initial_count, cascade_size, sum(obj.everActivated), obj.N, obj.current_step);
        end

        function external_count = getExternallyActivatedCount(obj)
            % 获取当前外源激活个体数量
            external_count = sum(obj.isExternallyActivated);
        end

        function external_indices = getExternallyActivatedIndices(obj)
            % 获取外源激活个体的索引
            external_indices = find(obj.isExternallyActivated);
        end
    end
end