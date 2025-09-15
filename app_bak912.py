import os
import gradio as gr
import requests
import json
from funasr import AutoModel
import torch
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 环境变量配置
GLM45_API_KEY = os.getenv("GLM45_API_KEY", "72c34a2761594dfeb6c9eb501f5c7c11.Wzgr6C7xNI7tqlZq")
GLM45_ENDPOINT = os.getenv("GLM45_ENDPOINT", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
GLM45V_API_KEY = os.getenv("GLM45V_API_KEY", "45897563484447298c2ff4441bb51b34.3mi3ExGuGuPP6i8a")
GLM45V_ENDPOINT = os.getenv("GLM45V_ENDPOINT", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
FUNASR_MODEL_DIR = os.getenv("FUNASR_MODEL_DIR", "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
USE_GPU = os.getenv("USE_GPU", "True").lower() == "true"

# 初始化FunASR模型
funasr_model = None

def initialize_funasr_model():
    """初始化FunASR模型"""
    global funasr_model
    if funasr_model is None:
        try:
            funasr_model = AutoModel(
                model=FUNASR_MODEL_DIR, 
                vad_model="fsmn-vad", 
                punc_model="ct-punc", 
                device="cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
            )
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("FunASR模型初始化后已清理GPU内存缓存")
        except Exception as e:
            print(f"FunASR模型初始化失败: {str(e)}")
            funasr_model = None

def transcribe_audio(audio_file):
    """使用FunASR进行语音识别"""
    if audio_file is None:
        print("未提供音频文件")
        return ""
    
    # 初始化模型（如果尚未初始化）
    print("正在初始化FunASR模型...")
    initialize_funasr_model()
    
    if funasr_model is None:
        print("FunASR模型初始化失败")
        return "FunASR模型未初始化"
    
    try:
        print(f"开始语音识别: {audio_file}")
        res = funasr_model.generate(input=audio_file, batch_size_s=300, hotword=" ")
        transcription = res[0]["text"]
        print(f"语音识别完成，识别结果: {transcription}")
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("语音识别完成后已清理GPU内存缓存")
        
        return transcription
    except Exception as e:
        error_msg = f"语音识别出错: {str(e)}"
        print(error_msg)
        # 即使出错也尝试清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return error_msg

def analyze_video(video_file, task_context):
    """使用MiniCPM-V-4_5分析视频内容"""
    print("开始分析视频内容...")
    if video_file is None:
        print("未提供视频文件")
        return "未提供视频文件"
    
    # 导入新的视频分析模块
    try:
        print("正在导入视频分析模块...")
        from minicpm_video_analyzer import analyze_video_with_minicpm
        print("正在调用MiniCPM-V-4_5模型进行视频内容分析...")
        result = analyze_video_with_minicpm(video_file, task_context)
        print("视频内容分析完成")
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        print("已清理GPU内存缓存")
        
        return result
    except Exception as e:
        error_msg = f"视频分析出错: {str(e)}"
        print(error_msg)
        # 即使出错也尝试清理内存
        torch.cuda.empty_cache()
        return error_msg

def analyze_audio(audio_file, task_context):
    """使用GLM-4.5分析音频转录内容"""
    print("开始分析音频内容...")
    if audio_file is None:
        print("未提供音频文件")
        return "未提供音频文件"
    
    # 先进行语音识别
    print("正在进行语音识别...")
    transcription = transcribe_audio(audio_file)
    print(f"语音识别结果: {transcription}")
    
    prompt = f"""请分析以下销售演讲音频转录文本，并结合任务背景提供评分建议。

任务背景：{task_context}

音频转录：{transcription}

请从以下几个维度进行评分（满分10分）：
1. 表达能力
2. 内容结构
3. 说服力
4. 声音语调

请提供详细的评分理由和改进建议。"""
    
    print("正在调用GLM-4.5模型进行音频内容分析...")
    headers = {
        "Authorization": f"Bearer {GLM45_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "glm-4.5",
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的销售演讲评估专家。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.6
    }
    
    try:
        response = requests.post(GLM45_ENDPOINT, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        print("音频内容分析完成")
        
        # 清理GPU内存（如果使用了GPU）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("已清理GPU内存缓存")
        
        return answer
    except Exception as e:
        error_msg = f"音频分析出错: {str(e)}"
        print(error_msg)
        # 即使出错也尝试清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return error_msg

def analyze_text(text_content, task_context):
    """使用GLM-4.5分析文本内容"""
    if not text_content.strip():
        return "未提供文本内容"
    
    prompt = f"""请分析以下销售演讲文稿，并结合任务背景提供评分建议。

任务背景：{task_context}

演讲文稿：{text_content}

请从以下几个维度进行评分（满分10分）：
1. 表达能力
2. 内容结构
3. 说服力

请提供详细的评分理由和改进建议。"""
    
    headers = {
        "Authorization": f"Bearer {GLM45_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "glm-4.5",
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的销售演讲评估专家。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.6
    }
    
    try:
        response = requests.post(GLM45_ENDPOINT, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        
        # 清理GPU内存（如果使用了GPU）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("文本分析完成后已清理GPU内存缓存")
        
        return answer
    except Exception as e:
        error_msg = f"文本分析出错: {str(e)}"
        print(error_msg)
        # 即使出错也尝试清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return error_msg

def score_presentation(video, audio, text, task_context):
    """主评分函数"""
    print("开始生成评分报告...")
    print(f"任务背景: {task_context}")
    
    print("正在分析视频内容...")
    video_result = analyze_video(video, task_context) if video else "未提供视频"
    print("视频分析完成")
    
    print("正在分析音频内容...")
    audio_result = analyze_audio(audio, task_context) if audio else "未提供音频"
    print("音频分析完成")
    
    print("正在分析文本内容...")
    text_result = analyze_text(text, task_context) if text else "未提供文本"
    print("文本分析完成")
    
    # 检查音频转录结果是否出错
    if "语音识别出错" in audio_result:
        audio_result = f"{audio_result}，请检查音频文件格式和内容。"
    
    print("正在生成综合评分报告...")
    # 综合评分报告
    full_report = f"""# 销售演讲评分报告

## 任务背景
{task_context}

## 视频分析结果
{video_result}

## 音频分析结果
{audio_result}

## 文本分析结果
{text_result}

## 综合建议
请结合以上各项分析结果，针对性地改进您的销售演讲技巧。"""
    
    print("评分报告生成完成")
    
    # 最终清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已清理GPU内存缓存")
    
    return full_report

# Gradio界面
description = """
# AI智能销售演讲评分系统

上传您的销售演讲视频、音频或文稿，并提供任务背景，系统将为您生成专业的评分报告。
"""

iface = gr.Interface(
    fn=score_presentation,
    inputs=[
        gr.Video(label="演讲视频"),
        gr.Audio(type="filepath", label="演讲音频"),
        gr.Textbox(label="演讲文稿", lines=5),
        gr.Textbox(label="任务背景", lines=3, placeholder="请描述本次销售演讲的任务背景，例如：向IT经理推销企业级软件解决方案")
    ],
    outputs=gr.Markdown(label="评分报告"),
    title="AI智能销售演讲评分系统",
    description=description,
    examples=[
        [None, None, "尊敬的客户，今天我将为您介绍我们公司的企业级软件解决方案...", "向IT经理推销企业级软件解决方案"]
    ]
)

if __name__ == "__main__":
    # 初始化FunASR模型
    initialize_funasr_model()
    iface.launch(server_name="0.0.0.0", server_port=7861, show_error=True)