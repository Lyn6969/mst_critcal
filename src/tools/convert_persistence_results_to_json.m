function convert_persistence_results_to_json(timestamp_dir)
% convert_persistence_results_to_json 将 persistence_results.mat 转写为 JSON
%
% 使用说明：
%   在 MATLAB 中运行：
%       convert_persistence_results_to_json('20251105_221002');
%   如果不传入参数，则默认读取 20251105_221002 这批实验数据。
%
% 输出：
%   在对应的 persistence_scan/<timestamp>/ 目录下生成 persistence_results.json，
%   其中包含元数据、坐标轴参数、统计结果矩阵以及原始样本，方便大模型解析。

    if nargin < 1 || isempty(timestamp_dir)
        timestamp_dir = '20251105_221002';
    end

    script_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(fileparts(script_dir));

    mat_file = 'persistence_results.mat';
    data_dir = fullfile(project_root, 'data', 'experiments', 'persistence_scan', timestamp_dir);
    mat_path = fullfile(data_dir, mat_file);

    if ~isfile(mat_path)
        error('未找到数据文件: %s', mat_path);
    end

    data = load(mat_path, 'results');
    if ~isfield(data, 'results')
        error('MAT 文件中缺少 results 结构体');
    end
    results = data.results;

    json_ready = struct();
    json_ready.description = results.description;
    json_ready.timestamp = results.timestamp;
    if isfield(results, 'generated_at')
        json_ready.generated_at = char(results.generated_at);
    else
        json_ready.generated_at = '';
    end
    json_ready.base_params = results.base_params;
    json_ready.config = results.config;

    axes_struct = struct();
    axes_struct.cj_thresholds = results.cj_thresholds;
    axes_struct.noise_levels = results.noise_levels;
    json_ready.axes = axes_struct;

    metrics_struct = struct();
    metrics_struct.D_mean = results.D_mean;
    metrics_struct.D_std = results.D_std;
    metrics_struct.P_mean = results.P_mean;
    metrics_struct.P_std = results.P_std;
    json_ready.metrics = metrics_struct;

    samples_struct = struct();
    if isfield(results, 'raw_D')
        samples_struct.raw_D = results.raw_D;
    else
        samples_struct.raw_D = [];
    end
    if isfield(results, 'raw_P')
        samples_struct.raw_P = results.raw_P;
    else
        samples_struct.raw_P = [];
    end
    json_ready.samples = samples_struct;

    json_text = jsonencode(json_ready, 'PrettyPrint', true);
    json_char = convertStringsToChars(json_text);

    output_path = fullfile(data_dir, 'persistence_results.json');
    fid = fopen(output_path, 'w');
    if fid == -1
        error('无法写入 JSON 文件: %s', output_path);
    end
    cleaner = onCleanup(@() fclose(fid));

    bytes_written = fwrite(fid, json_char, 'char');
    if bytes_written ~= numel(json_char)
        warning('写入的字节数 (%d) 少于预期 (%d)。', bytes_written, numel(json_char));
    end

    fprintf('JSON 文件已生成：%s\n', output_path);
end
