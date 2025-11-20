function fix_persistence_scaling(target_patterns, varargin)
% fix_persistence_scaling 针对既有持久性实验结果，批量修正二维扩散系数漏乘维度因子的历史数据。
%
% 用法示例：
%   fix_persistence_scaling({'results/persistence/*.mat', 'results/tradeoff/*.mat'});
%   fix_persistence_scaling('results/tradeoff/cj_tradeoff_adaptive_shared_seed_*.mat', 'DryRun', true);
%
% 参数说明：
%   target_patterns - 字符串或字符串元胞数组，可包含通配符，指向需要修正的 .mat 文件。
%   Name-Value 选项：
%       'Backup'       (logical, 默认 true)     修正前是否生成 .bak 副本。
%       'BackupSuffix' (char,    默认 '.pre_pfix') 备份文件后缀。
%       'DryRun'       (logical, 默认 false)     仅预览修改，不写回文件。
%       'Verbose'      (logical, 默认 true)      是否输出详细日志。
%       'DiffusionFloor' (double, 默认 1e-3)     D_* 保底下限，用于避免已裁剪值被继续缩小。
%       'RespectDiffusionFloor' (logical, 默认 true)
%            为 true 时，仅对大于下限的 D_* 乘 0.25，并把小于等于下限的值锁定在下限；
%            为 false 时，对所有 D_* 值一律乘 0.25（旧行为）。
%       'CapPersistenceAtFloor' (logical, 默认 true)
%            为 true 且启用下限保护时，P_* 放大 2 倍后会以 1/sqrt(DiffusionFloor) 为上限，避免下限处被意外翻倍。
%
% 注意：脚本仅会处理包含变量 "results" 或 "summary" 的 .mat 文件，
%       其他变量（如 cfg、params 等）会被忽略。每个数据集只会修正一次，
%       并写入 persistence_rescale_info 元数据，后续再次运行会自动跳过。

    if nargin < 1 || isempty(target_patterns)
        error('请至少提供一个文件路径或通配符。');
    end

    patterns = normalize_patterns(target_patterns);

    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'Backup', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'BackupSuffix', '.pre_pfix', @(x) ischar(x) || (isstring(x) && isscalar(x)));
    addParameter(parser, 'DryRun', false, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'Verbose', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'DiffusionFloor', 1e-3, @(x) isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(parser, 'RespectDiffusionFloor', true, @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'CapPersistenceAtFloor', true, @(x) islogical(x) && isscalar(x));
    parse(parser, varargin{:});
    opts = parser.Results;

    file_list = expand_file_list(patterns);
    if isempty(file_list)
        warning('未找到匹配的 .mat 文件。');
        return;
    end

    fprintf('将处理 %d 个文件（DryRun=%d, Backup=%d）。\n', numel(file_list), opts.DryRun, opts.Backup);

    for idx = 1:numel(file_list)
        mat_path = file_list{idx};
        data = load(mat_path);
        [data, stats] = patch_dataset_variables(data, opts);

        if stats.already_fixed
            if opts.Verbose
                fprintf('[%02d/%02d] %s 已含 persistence_rescale_info，跳过。\n', ...
                    idx, numel(file_list), mat_path);
            end
            continue;
        end

        if ~(stats.changed)
            if opts.Verbose
                fprintf('[%02d/%02d] %s 不包含可修正字段，跳过。\n', idx, numel(file_list), mat_path);
            end
            continue;
        end

        if opts.DryRun
            fprintf('[%02d/%02d] %s 需修正：P字段 %d 个，D字段 %d 个（DryRun不写盘）。\n', ...
                idx, numel(file_list), mat_path, stats.p_fields, stats.d_fields);
            continue;
        end

        if opts.Backup
            backup_path = strcat(mat_path, opts.BackupSuffix);
            copyfile(mat_path, backup_path);
        end

        save(mat_path, '-struct', 'data', '-v7.3');
        fprintf('[%02d/%02d] %s 修正完成：P字段 %d 个，D字段 %d 个。\n', ...
            idx, numel(file_list), mat_path, stats.p_fields, stats.d_fields);
    end
end

function patterns = normalize_patterns(raw_patterns)
    if ischar(raw_patterns) || isstring(raw_patterns)
        raw_patterns = cellstr(raw_patterns);
    elseif ~iscell(raw_patterns)
        error('target_patterns 需要是字符向量、字符串或元胞数组。');
    end
    patterns = raw_patterns(:);
end

function files = expand_file_list(patterns)
    files = {};
    for i = 1:numel(patterns)
        pat = char(patterns{i});
        listing = dir(pat);
        for k = 1:numel(listing)
            if listing(k).isdir
                continue;
            end
            files{end+1} = fullfile(listing(k).folder, listing(k).name); %#ok<AGROW>
        end
    end
    files = unique(files);
end

function [data, stats] = patch_dataset_variables(data, opts)
    stats = struct('changed', false, 'already_fixed', false, 'p_fields', 0, 'd_fields', 0);
    target_vars = intersect(fieldnames(data), {'results', 'summary'});
    if isempty(target_vars)
        return;
    end

    for i = 1:numel(target_vars)
        name = target_vars{i};
        dataset = data.(name);
        if ~isstruct(dataset)
            continue;
        end
        if isfield(dataset, 'persistence_rescale_info')
            stats.already_fixed = true;
            continue;
        end

        [dataset, patch_stats] = rescale_struct_recursive(dataset, opts);
        if patch_stats.changed
            dataset.persistence_rescale_info = struct( ...
                'version', 1, ...
                'dimension', 2, ...
                'p_factor', 2.0, ...
                'd_factor', 0.25, ...
                'applied_at', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
            data.(name) = dataset;
            stats.changed = true;
            stats.p_fields = stats.p_fields + patch_stats.p_fields;
            stats.d_fields = stats.d_fields + patch_stats.d_fields;
        end
    end
end

function [s, stats] = rescale_struct_recursive(s, opts)
    stats = struct('changed', false, 'p_fields', 0, 'd_fields', 0);
    field_names = fieldnames(s);
    for i = 1:numel(field_names)
        key = field_names{i};
        value = s.(key);

        if isstruct(value)
            for idx = 1:numel(value)
                [value(idx), child_stats] = rescale_struct_recursive(value(idx), opts);
                if child_stats.changed
                    s.(key) = value;
                    stats.changed = true;
                    stats.p_fields = stats.p_fields + child_stats.p_fields;
                    stats.d_fields = stats.d_fields + child_stats.d_fields;
                end
            end
            continue;
        end

        if ~(isnumeric(value) || islogical(value)) || isempty(value)
            continue;
        end

        if should_scale_persistence(key)
            origValP = double(value);
            newValP = origValP;
            changed_local = false;
            if opts.RespectDiffusionFloor && opts.CapPersistenceAtFloor
                p_cap = 1 / sqrt(max(opts.DiffusionFloor, eps));
                mask = origValP < p_cap - eps(p_cap);  % 仅对未达到上限的值放大
                if any(mask, 'all')
                    newValP(mask) = min(origValP(mask) * 2.0, p_cap);
                    changed_local = any(newValP(mask) ~= origValP(mask), 'all');
                end
            else
                newValP = origValP * 2.0;
                changed_local = ~isequaln(newValP, origValP);
            end
            if changed_local
                s.(key) = newValP;
                stats.changed = true;
                stats.p_fields = stats.p_fields + 1;
            end
        elseif should_scale_diffusion(key)
            origValD = double(value);
            newValD = origValD;
            changed_local = false;
            if opts.RespectDiffusionFloor
                floor_val = opts.DiffusionFloor;
                % 仅对超过下限的值执行 0.25 缩放，已在下限的保持不变
                mask = origValD > floor_val + eps(floor_val);
                if any(mask, 'all')
                    newValD(mask) = origValD(mask) * 0.25;
                    changed_local = true;
                end
                % 对低于下限的数值拉升到下限
                below_mask = origValD < floor_val - eps(floor_val);
                if any(below_mask, 'all')
                    newValD(below_mask) = floor_val;
                    changed_local = true;
                end
            else
                newValD = origValD * 0.25;
                changed_local = ~isequaln(newValD, origValD);
            end
            if changed_local
                s.(key) = newValD;
                stats.changed = true;
                stats.d_fields = stats.d_fields + 1;
            end
        end
    end
end

function tf = should_scale_persistence(field_name)
    if ~startsWith(field_name, 'P_')
        tf = false;
        return;
    end
    tf = ~contains(lower(field_name), 'norm');
end

function tf = should_scale_diffusion(field_name)
    tf = startsWith(field_name, 'D_');
end
