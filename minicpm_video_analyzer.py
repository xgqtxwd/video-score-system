import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
from scipy.spatial import cKDTree
import numpy as np
import math
import base64
import os

# 设置PyTorch内存优化选项
torch.backends.cudnn.benchmark = False  # 对于动态输入大小，设置为False可以减少内存使用
torch.backends.cudnn.deterministic = True  # 确保结果可重现
# 设置PyTorch内存分配器选项以优化内存使用
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # 减少内存碎片

# 模型路径配置
MINICPM_MODEL_PATH = "/home/models/MiniCpm/OpenBMB/MiniCPM-V-4_5/"

# 初始化模型
print("正在初始化MiniCPM-V-4_5模型...")
try:
    model = AutoModel.from_pretrained(
        MINICPM_MODEL_PATH, 
        trust_remote_code=True, 
        attn_implementation='sdpa', 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # 减少CPU内存使用
        device_map="auto"  # 自动分配设备以优化内存使用
    )
    # 检查是否有多个GPU可用
    if torch.cuda.device_count() > 1:
        print(f"发现 {torch.cuda.device_count()} 个GPU，使用数据并行处理")
        model = torch.nn.DataParallel(model)
        # 保存原始模型引用以便调用chat方法
        original_model = model.module
    else:
        model = model.eval()  # 如果只有一个GPU，不需要调用cuda()，因为device_map会自动处理
        original_model = model
    print("MiniCPM-V-4_5模型初始化完成")
except Exception as e:
    print(f"模型初始化失败: {str(e)}")
    # 尝试使用更节省内存的方式加载
    print("尝试加载模型...")
    model = AutoModel.from_pretrained(
        MINICPM_MODEL_PATH, 
        trust_remote_code=True, 
        attn_implementation='sdpa', 
        torch_dtype=torch.float16,  # 使用float16而不是bfloat16以节省内存
        low_cpu_mem_usage=True,
        device_map="auto"  # 自动分配设备以优化内存使用
    )
    # 检查是否有多个GPU可用
    if torch.cuda.device_count() > 1:
        print(f"发现 {torch.cuda.device_count()} 个GPU，使用数据并行处理")
        model = torch.nn.DataParallel(model)
        # 保存原始模型引用以便调用chat方法
        original_model = model.module
    else:
        model = model.eval()  # 如果只有一个GPU，不需要调用cuda()，因为device_map会自动处理
        original_model = model
    print("MiniCPM-V-4_5模型初始化完成(使用float16)")

tokenizer = AutoTokenizer.from_pretrained(MINICPM_MODEL_PATH, trust_remote_code=True)

MAX_NUM_FRAMES = 50  # 进一步减少最大帧数以降低内存使用
MAX_NUM_PACKING = 1  # 减少打包数量
TIME_SCALE = 0.1


def map_to_nearest_scale(values, scale):
    tree = cKDTree(np.asarray(scale)[:, None])
    _, indices = tree.query(np.asarray(values)[:, None])
    return np.asarray(scale)[indices]


def group_array(arr, size):
    return [arr[i:i+size] for i in range(0, len(arr), size)]


def encode_video(video_path, choose_fps=1, force_packing=None):
    print(f"开始编码视频: {video_path}, choose_fps={choose_fps}")
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    video_duration = len(vr) / fps
    print(f"视频信息 - FPS: {fps}, 时长: {video_duration}秒")
        
    if choose_fps * int(video_duration) <= MAX_NUM_FRAMES:
        packing_nums = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(MAX_NUM_FRAMES, video_duration))
        
    else:
        packing_nums = math.ceil(video_duration * choose_fps / MAX_NUM_FRAMES)
        if packing_nums <= MAX_NUM_PACKING:
            choose_frames = round(video_duration * choose_fps)
        else:
            choose_frames = round(MAX_NUM_FRAMES * MAX_NUM_PACKING)
            packing_nums = MAX_NUM_PACKING

    frame_idx = [i for i in range(0, len(vr))]
    frame_idx = np.array(uniform_sample(frame_idx, choose_frames))

    if force_packing:
        packing_nums = min(force_packing, MAX_NUM_PACKING)
    
    print(video_path, ' duration:', video_duration)
    print(f'采样帧数={len(frame_idx)}, 打包数量={packing_nums}')
    
    # 分批读取帧以减少内存峰值
    BATCH_SIZE = 16  # 每批读取的帧数
    frames = []
    for i in range(0, len(frame_idx), BATCH_SIZE):
        batch_idx = frame_idx[i:i+BATCH_SIZE]
        batch_frames = vr.get_batch(batch_idx).asnumpy()
        frames.extend(batch_frames)
        # 及时清理GPU内存
        torch.cuda.empty_cache()

    frame_idx_ts = frame_idx / fps
    scale = np.arange(0, video_duration, TIME_SCALE)

    frame_ts_id = map_to_nearest_scale(frame_idx_ts, scale) / TIME_SCALE
    frame_ts_id = frame_ts_id.astype(np.int32)

    assert len(frames) == len(frame_ts_id)

    frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
    frame_ts_id_group = group_array(frame_ts_id, packing_nums)
    
    print(f"视频编码完成，返回 {len(frames)} 帧和 {len(frame_ts_id_group)} 个时间组")
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    print("视频编码完成后已清理GPU内存缓存")
    
    return frames, frame_ts_id_group


def analyze_video_with_minicpm(video_file, task_context, scoring_prompt=None):
    """使用MiniCPM-V-4_5分析视频内容"""
    print(f"开始分析视频: {video_file}")
    if video_file is None:
        print("未提供视频文件")
        return "未提供视频文件"
    
    try:
        # 编码视频
        print("正在编码视频...")
        frames, frame_ts_id_group = encode_video(video_file, choose_fps=1)  # 进一步降低帧率以减少内存使用
        print(f"视频编码完成，共获取到 {len(frames)} 帧")
        
        # 使用传入的评分标准构建提示词，如果没有则使用默认提示词
        if scoring_prompt:
            prompt = f"""请根据以下评分标准分析销售演讲视频，并结合任务背景提供评分建议。

任务背景：{task_context}

评分标准：{scoring_prompt}

请严格按照评分标准进行分析和评分，并提供详细的评分理由和改进建议。"""
        else:
            # 构造提示词，专注于销售人员的面部表情和肢体动作分析
            prompt = f"""请分析以下销售演讲视频，并结合任务背景提供评分建议。

任务背景：{task_context}

请从以下几个维度进行评分（满分10分）：
1. 面部表情自然度 - 销售人员的面部表情是否自然、真诚，是否与演讲内容相匹配
2. 眼神交流 - 销售人员的眼神是否稳定，是否有飘忽不定的情况，是否与镜头有良好的交流
3. 肢体语言 - 销售人员的肢体动作是否自然、得体，是否有助于表达演讲内容
4. 整体表现力 - 销售人员的整体表现是否自信、专业，是否能够吸引观众的注意力

请提供详细的评分理由和改进建议。"""
        
        # 构造消息
        msgs = [
            {'role': 'user', 'content': frames + [prompt]},
        ]
        
        # 调用模型
        print("正在调用MiniCPM-V-4_5模型进行视频分析...")
        answer = original_model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            use_image_id=False,
            max_slice_nums=1,
            temporal_ids=frame_ts_id_group,
            max_new_tokens=512,  # 进一步限制生成的最大token数以节省内存
            device="cuda" if torch.cuda.is_available() else "cpu"  # 根据可用设备选择
        )
        
        print("视频分析完成")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        print("已清理GPU内存缓存")
        
        return answer
    
    except Exception as e:
        error_msg = f"视频分析出错: {str(e)}"
        print(error_msg)
        # 即使出错也尝试清理内存
        torch.cuda.empty_cache()
        return error_msg