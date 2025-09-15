# AI智能销售演讲评分系统

## 项目简介

AI智能销售演讲评分系统是一个基于人工智能技术的自动评分工具，专门用于评估销售演讲视频的表现。该系统能够分析演讲者的面部表情、肢体动作、语言表达等多个维度，并提供专业的评分和改进建议。

## 核心功能

1. **视频内容分析**：使用MiniCPM-V-4_5视觉模型分析演讲者的面部表情、眼神交流和肢体语言
2. **音频内容分析**：通过FunASR语音识别技术转录音频，并使用GLM-4.5大语言模型分析语言表达能力
3. **自定义评分标准**：支持上传DOCX格式的自定义评分标准文件
4. **综合评分报告**：生成包含视频和音频分析结果的综合评分报告
5. **报告导出**：支持导出Markdown和PDF格式的详细报告

## 技术架构

### 核心组件

- **MiniCPM-V-4_5**：用于视频内容分析的视觉模型
- **FunASR**：用于语音识别的音频处理工具
- **GLM-4.5**：用于文本分析的大语言模型
- **LangGraph**：用于工作流管理的状态图框架
- **Gradio**：用于构建Web界面的机器学习应用框架

### 工作流程

1. 用户上传销售演讲视频和任务背景描述
2. 系统从视频中提取音频
3. 使用FunASR进行音频转录
4. 使用MiniCPM-V-4_5分析视频内容（面部表情、肢体动作等）
5. 使用GLM-4.5分析音频内容（语言表达、销售技巧等）
6. 根据自定义评分标准或默认标准生成综合评分报告
7. 提供Markdown和PDF格式的报告下载

## 安装与配置

### 环境要求

- Python 3.8+
- CUDA 11.8+（用于GPU加速，可选）
- FFmpeg（用于音视频处理）

### 安装步骤

1. 克隆项目代码：
   ```bash
   git clone <repository-url>
   cd video-score-system
   ```

2. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

3. 安装FFmpeg：
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # 从 https://ffmpeg.org/download.html 下载并安装
   ```

### 环境变量配置

在项目根目录创建 `.env` 文件，配置以下环境变量：

```env
# GLM-4.5 API配置
GLM45_API_KEY=your_api_key_here
GLM45_ENDPOINT=https://open.bigmodel.cn/api/paas/v4/chat/completions

# FunASR模型配置
FUNASR_MODEL_DIR=iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
USE_GPU=True
```

### 模型配置

1. **MiniCPM-V-4_5模型**：
   - 下载模型到本地路径（默认：`/home/models/MiniCpm/OpenBMB/MiniCPM-V-4_5/`）
   - 修改 `minicpm_video_analyzer.py` 中的 `MINICPM_MODEL_PATH` 变量指向模型路径

## 使用方法

1. 启动应用：
   ```bash
   python app.py
   ```

2. 在浏览器中访问 `http://localhost:7861`

3. 上传销售演讲视频文件

4. 输入任务背景描述

5. （可选）上传自定义评分标准DOCX文件

6. 点击提交，等待系统生成评分报告

7. 下载Markdown或PDF格式的详细报告

## 自定义评分标准

系统支持自定义评分标准，用户可以上传DOCX格式的评分标准文件。文件中应详细描述评分维度、评分标准和权重分配。

如果没有提供自定义评分标准，系统将使用默认评分标准：
1. 面部表情及肢体动作表现（25分）
2. 语言表现能力（25分）
3. 内容组织与呈现（25分）
4. 专业性与说服力（25分）

## 项目结构

```
video-score-system/
├── app.py                 # 主应用文件，包含Gradio界面
├── workflow.py            # LangGraph工作流实现
├── minicpm_video_analyzer.py  # MiniCPM-V-4_5视频分析模块
├── requirements.txt       # 项目依赖
├── .env                  # 环境变量配置文件
├── README.md             # 项目说明文档
└── .gitignore            # Git忽略文件配置
```

## API接口

### 视频分析接口

```python
analyze_video_with_minicpm(video_file, task_context, scoring_prompt=None)
```

参数：
- `video_file`: 视频文件路径
- `task_context`: 任务背景描述
- `scoring_prompt`: 评分标准（可选）

### 音频分析接口

```python
analyze_audio(audio_file, task_context, scoring_prompt=None)
```

参数：
- `audio_file`: 音频文件路径
- `task_context`: 任务背景描述
- `scoring_prompt`: 评分标准（可选）

## 性能优化

1. **内存管理**：系统实现了GPU内存的自动清理机制
2. **视频处理优化**：通过降低采样帧率和分批处理来减少内存使用
3. **模型加载优化**：使用`low_cpu_mem_usage`和`device_map="auto"`优化模型加载

## 故障排除

### 常见问题

1. **模型加载失败**：
   - 检查模型路径是否正确
   - 确认模型文件是否完整
   - 检查GPU内存是否充足

2. **音频提取失败**：
   - 确认FFmpeg是否正确安装
   - 检查视频文件格式是否支持

3. **API调用失败**：
   - 检查网络连接
   - 确认API密钥是否正确

### 日志查看

系统会在控制台输出详细的处理日志，可以通过日志定位问题。

## 许可证

本项目采用MIT许可证，详情请参见LICENSE文件。

## 联系方式

如有问题或建议，请联系项目维护者。