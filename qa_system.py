import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import psutil  # 用于检查系统内存
from modelscope import snapshot_download

# 定义固定模型存储路径
BASE_MODEL_DIR = "D:/code/ragflow_practice/models"

# 提前下载模型
def download_models_if_needed():
    # 1. 中文嵌入模型
    embedding_model_dir = os.path.join(BASE_MODEL_DIR, "bge-small-zh-v1.5")
    if not os.path.exists(embedding_model_dir):
        print(f"正在下载中文嵌入模型到 {embedding_model_dir}...")
        snapshot_download(
            'Xorbits/bge-small-zh-v1.5',
            local_dir=embedding_model_dir
        )
    
    # 2. 医疗专用问答模型
    qa_model_dir = os.path.join(BASE_MODEL_DIR, "CareBot_Medical")
    if not os.path.exists(qa_model_dir):
        print(f"正在下载医疗专用问答模型到 {qa_model_dir}...")
        snapshot_download(
            'BAAI/CareBot_Medical_multi-llama3-8b-instruct',
            local_dir=qa_model_dir
        )
    
    return qa_model_dir, embedding_model_dir

# 检查系统内存和虚拟内存
def check_system_memory():
    virtual_memory = psutil.virtual_memory()
    total_mem_gb = virtual_memory.total / (1024 **3)
    available_mem_gb = virtual_memory.available / (1024** 3)
    
    # 检查物理内存是否充足（8B模型建议至少16GB）
    print(f"\n系统内存检查：总内存 {total_mem_gb:.1f}GB，可用 {available_mem_gb:.1f}GB")
    if total_mem_gb < 16:
        print("警告：物理内存不足16GB，可能导致模型运行失败")
        print("建议：增加物理内存或扩大虚拟内存（页面文件）")
        print("虚拟内存设置方法：此电脑→属性→高级系统设置→性能→高级→虚拟内存→自定义大小（建议设为物理内存的1.5-3倍）")
    
    # 检查可用内存
    if available_mem_gb < 8:
        print("警告：可用内存不足8GB，建议关闭其他程序释放内存")
        input("按回车键继续（确保已关闭其他程序）...")

# 获取模型路径
local_qa_model, local_embedding_model = download_models_if_needed()

# 配置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def main():
    try:
        # 检查系统内存
        check_system_memory()

        # 检查模型是否存在
        if not os.path.exists(local_embedding_model) or not os.path.exists(local_qa_model):
            print(f"错误：模型未在 {BASE_MODEL_DIR} 中找到，请重新下载")
            return

        # 初始化Chroma客户端
        chroma_client = chromadb.PersistentClient(path="./chroma_data")

        # 连接集合（GPU加速）
        try:
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=local_embedding_model,
                device="cuda"
            )
            collection = chroma_client.get_collection(
                name="knowledges",
                embedding_function=embedding_func
            )
            print("成功连接到向量集合（GPU加速嵌入）")
        except Exception as e:
            print(f"连接向量集合失败：{str(e)}")
            print("请先运行document_processer.py生成集合")
            return

        # 加载医疗问答模型（内存优化版）
        try:
            print("加载医疗专用问答模型（内存优化模式）...")
            # 确保accelerate已安装
            import accelerate
            
            # 检查是否安装bitsandbytes（用于量化）
            try:
                import bitsandbytes
                use_quantization = True
                print("检测到bitsandbytes，将使用4位量化减少内存占用")
            except ImportError:
                use_quantization = False
                print("未检测到bitsandbytes，尝试安装以启用内存优化...")
                os.system("pip install bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple")
                try:
                    import bitsandbytes
                    use_quantization = True
                except ImportError:
                    print("bitsandbytes安装失败，将使用8位量化")
                    use_quantization = False

            tokenizer = AutoTokenizer.from_pretrained(local_qa_model)
            
            # 根据内存情况选择量化方式
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": "auto"
            }
            
            if use_quantization:
                model_kwargs["load_in_4bit"] = True  # 4位量化（最省内存）
            else:
                model_kwargs["load_in_8bit"] = True  # 8位量化（兼容性更好）

            model = AutoModelForCausalLM.from_pretrained(
                local_qa_model,** model_kwargs
            )
            
            # 限制生成长度，减少内存占用
            qa_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=200,  # 减少生成长度，降低内存需求
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.15
            )
            print("成功加载医疗专用问答模型（BAAI/CareBot）")
        except Exception as e:
            print(f"加载问答模型失败：{str(e)}")
            print("最后尝试方案：使用CPU运行（无量化，内存需求高）")
            # 备用方案：纯CPU运行（无量化）
            try:
                tokenizer = AutoTokenizer.from_pretrained(local_qa_model)
                model = AutoModelForCausalLM.from_pretrained(
                    local_qa_model,
                    device_map="cpu",  # 强制使用CPU
                    torch_dtype="auto"
                )
                qa_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=150
                )
                print("已切换到CPU运行模式（速度较慢）")
            except Exception as e2:
                print(f"CPU模式也失败：{str(e2)}")
                print("请增加内存或虚拟内存后重试")
                return

        # 交互问答
        print("\n医疗问答系统已启动（输入 'exit' 退出）")
        while True:
            question = input("\n请输入医疗问题：")
            if question.lower() == "exit":
                print("再见！")
                break

            # 检索相关文档
            try:
                results = collection.query(query_texts=[question], n_results=3)
                context = "\n\n".join(results["documents"][0])
                if not context:
                    print("未找到相关医疗信息")
                    continue
            except Exception as e:
                print(f"检索失败：{str(e)}")
                continue

            # 构造提示词
            prompt = f"""你是专业的医疗顾问，需根据以下医疗知识回答问题。
            医疗知识：{context}
            问题：{question}
            请用简洁、准确的中文回答，基于提供的医疗知识，避免猜测。"""

            # 生成回答
            try:
                print("生成回答中...（内存占用较高，请勿关闭程序）")
                result = qa_pipeline(prompt)[0]["generated_text"]
                answer = result.replace(prompt, "").strip()
                print(f"回答：{answer}")
            except Exception as e:
                print(f"生成回答失败：{str(e)}")
                print("可能是内存不足导致，建议：")
                print("1. 关闭所有其他程序")
                print("2. 扩大虚拟内存")
                print("3. 重启电脑后重试")
                continue

    except Exception as e:
        print(f"程序运行出错：{str(e)}")

if __name__ == "__main__":
    main()
    