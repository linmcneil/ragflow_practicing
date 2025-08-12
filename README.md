# 中文医疗问答系统（基于RAG技术）

这是一个基于检索增强生成（RAG）技术的中文医疗问答系统，能够利用医疗数据集回答用户的健康问题。系统结合了向量检索和医疗专用大模型，确保回答的准确性和专业性。

## 项目功能

- 从医疗数据集中提取知识并构建向量数据库
- 使用GPU加速模型推理，提升响应速度
- 针对医疗领域优化的问答逻辑，支持常见疾病咨询
- 内存优化机制，适配不同硬件配置

## 环境要求

- Python 3.8+
- 推荐配置：
  - CPU：4核及以上
  - 内存：16GB+（越大越好）
  - GPU：NVIDIA显卡（8GB显存以上，支持CUDA）
  - 硬盘：至少50GB空闲空间（用于存储模型和数据）

## 安装步骤

1. 克隆项目到本地
   ```bash
   git clone https://github.com/limmcneil/ragflow_practicing.git
   cd ragflow_practicing
   ```

2. 安装依赖包
   ```bash
   # 基础依赖
   pip install chromadb transformers modelscope sentence-transformers psutil -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   # 加速库（GPU必需）
   pip install accelerate bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   # 若使用GPU，需安装带CUDA的PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. 配置虚拟内存（重要）
   - 右键"此电脑"→"属性"→"高级系统设置"→"性能"→"高级"→"虚拟内存"
   - 建议设置为物理内存的1.5-3倍（如16GB物理内存→24-48GB虚拟内存）


## 使用方法

### 1. 预处理数据（首次运行必需）

运行数据处理脚本，将医疗数据集转换为向量数据库：python document_processer.py- 脚本会自动下载医疗数据集（若未存在）
- 对数据进行分块处理并生成向量
- 向量数据会存储在`chroma_data/`目录

### 2. 启动问答系统
python qa_system.py- 首次运行会自动下载所需模型（约15GB）
- 系统启动后，输入医疗问题即可获得回答
- 输入"exit"退出系统

## 常见问题

### 1. 模型下载缓慢
- 确保已配置国内镜像：`os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"`
- 可手动下载模型并放入`models/`目录对应文件夹

### 2. 内存不足错误（页面文件太小）
- 按照"安装步骤3"扩大虚拟内存
- 关闭其他占用内存的程序
- 重启电脑后重试

### 3. GPU加速未生效
- 检查CUDA环境是否配置正确
- 确认PyTorch安装的是带CUDA的版本
- 查看错误日志，确认是否有GPU相关报错

### 4. 回答准确率低
- 确保已正确生成向量数据库（运行`document_processer.py`）
- 尝试增加检索数量（修改`n_results`参数）

## 注意事项

- 本系统仅提供参考信息，不能替代专业医疗建议
- 模型运行时会占用大量系统资源，建议关闭其他程序
- 首次运行会下载较大模型文件，需耐心等待
- 若硬件配置较低，可尝试降低模型量化精度或使用CPU模式

## 技术细节

- 嵌入模型：`Xorbits/bge-small-zh-v1.5`（中文语义理解优化）
- 问答模型：`BAAI/CareBot_Medical_multi-llama3-8b-instruct`（医疗领域专用）
- 向量数据库：Chroma（轻量级向量存储）
- 检索增强：从向量库获取相关医疗知识作为上下文，提升回答准确性

---

如有任何问题或建议，欢迎提交issue或联系开发者。

这是本人第一个项目，有任何做的不周到的地方请大家海涵。
    
