import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import os
from modelscope import snapshot_download

# 提前下载模型
def download_models_if_needed():
    qa_model_dir = os.path.join(os.getcwd(), "models", "nlp_structbert_faq")
    embedding_model_dir = os.path.join(os.getcwd(), "models", "bge-small-zh-v1.5")
    
    if not os.path.exists(qa_model_dir):
        print("正在下载中文问答模型...")
        snapshot_download('iic/nlp_structbert_faq-question-answering_chinese-base', local_dir=qa_model_dir)
    
    if not os.path.exists(embedding_model_dir):
        print("正在下载中文嵌入模型...")
        snapshot_download('Xorbits/bge-small-zh-v1.5', local_dir=embedding_model_dir)
    
    return qa_model_dir, embedding_model_dir

# 获取模型路径
local_qa_model, local_embedding_model = download_models_if_needed()

# 配置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def main():
    try:
        # 检查模型是否存在
        if not os.path.exists(local_embedding_model) or not os.path.exists(local_qa_model):
            print("错误：模型文件不完整，请重新下载")
            return

        # 关键：使用磁盘存储的Chroma客户端（与document_processer.py对应）
        chroma_client = chromadb.PersistentClient(path="./chroma_data")

        # 连接集合（此时集合会从磁盘读取，不会丢失）
        try:
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=local_embedding_model,
                device="cpu"
            )
            collection = chroma_client.get_collection(
                name="knowledges",
                embedding_function=embedding_func
            )
            print("成功连接到向量集合")
        except Exception as e:
            print(f"连接向量集合失败：{str(e)}")
            print("请先运行document_processer.py生成集合")
            return

        # 加载问答模型
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_qa_model)
            model = AutoModelForQuestionAnswering.from_pretrained(local_qa_model)
            qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)
            print("成功加载中文问答模型")
        except Exception as e:
            print(f"加载问答模型失败：{str(e)}")
            return

        # 交互问答
        print("\n中文医疗问答系统已启动（输入 'exit' 退出）")
        while True:
            question = input("\n请输入问题：")
            if question.lower() == "exit":
                print("再见！")
                break

            # 检索相关文档
            try:
                results = collection.query(query_texts=[question], n_results=3)
                context = "\n\n".join(results["documents"][0])
                if not context:
                    print("未找到相关信息")
                    continue
            except Exception as e:
                print(f"检索失败：{str(e)}")
                continue

            # 生成回答
            try:
                result = qa_pipeline(question=question, context=context)
                print(f"回答：{result['answer']}")
                print(f"置信度：{round(result['score'], 4)}")
            except Exception as e:
                print(f"生成回答失败：{str(e)}")
                continue

    except Exception as e:
        print(f"程序运行出错：{str(e)}")

if __name__ == "__main__":
    main()