from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import os
import re  # 用于中文语义分块
from modelscope import snapshot_download

# 解决模型下载问题：设置国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def main():
    try:
        # 1. 下载中文嵌入模型（Xorbits/bge-small-zh-v1.5）
        embedding_model_dir = os.path.join(os.getcwd(), "models", "bge-small-zh-v1.5")
        if not os.path.exists(embedding_model_dir):
            print("正在下载中文嵌入模型...")
            snapshot_download(
                model_id='Xorbits/bge-small-zh-v1.5',
                local_dir=embedding_model_dir
            )

        # 2. 检查PDF文件是否存在
        pdf_path = "medical_guide.pdf"  # 你的PDF文件名
        try:
            pdf_reader = PdfReader(pdf_path)
            print(f"成功加载PDF文件，共{len(pdf_reader.pages)}页")
        except FileNotFoundError:
            print(f"错误：未找到PDF文件 '{pdf_path}'，请检查路径是否正确")
            return

        # 3. 初始化Chroma客户端（持久化存储到磁盘）
        chroma_client = chromadb.PersistentClient(path="./chroma_data")

        # 4. 强制删除已存在的集合（避免重复）
        try:
            chroma_client.delete_collection(name="knowledges")
        except:
            pass

        # 5. 初始化嵌入函数（使用下载好的中文模型）
        try:
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_dir,  # 本地模型路径
                device="cpu"
            )
            print("嵌入函数初始化成功")
        except Exception as e:
            print(f"嵌入函数初始化失败：{str(e)}")
            return

        # 6. 创建集合
        try:
            collection = chroma_client.create_collection(
                name="knowledges",
                embedding_function=embedding_func
            )
            print("集合创建成功")
        except Exception as e:
            print(f"集合创建失败：{str(e)}")
            return

        # 7. 提取PDF文本
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                print(f"警告：第{page_num+1}页未提取到文本")

        if not text:
            print("错误：未从PDF中提取到任何文本")
            return
        
        # 调试：打印提取的文本（可选，确认内容是否正确）
        # print("提取的文本内容：\n", text[:500], "...")  # 只打印前500字

        # 8. 优化的中文分块策略（针对医疗文本）
        # 步骤1：按段落拆分（连续空行为分隔符）
        paragraphs = re.split(r'\n\s*\n', text.strip())  # 处理PDF中的空行分隔
        chunks = []
        
        # 步骤2：对每个段落按语义拆分（保留完整句子和医疗术语）
        for para in paragraphs:
            if not para:
                continue
            
            # 清除多余空格和换行
            para_clean = re.sub(r'\s+', ' ', para).strip()
            
            # 按中文句尾符号拆分（。！？；），保留分隔符
            sentences = re.split(r'(。|！|？|；)', para_clean)
            
            # 合并句子，控制块大小（约200-300字，适合医疗文本）
            current_chunk = ""
            for sent in sentences:
                # 跳过空内容
                if not sent.strip():
                    continue
                
                # 检查当前块加上新句子后的长度
                if len(current_chunk) + len(sent) < 300:
                    current_chunk += sent
                else:
                    # 确保块以完整句尾结束
                    if current_chunk.endswith(('。', '！', '？', '；')):
                        chunks.append(current_chunk.strip())
                        current_chunk = sent
                    else:
                        # 若未结束，强制分割并补全句尾
                        current_chunk += sent
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
            
            # 添加最后一个不完整的块
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

        # 去重（避免重复内容）
        chunks = list(dict.fromkeys(chunks))
        print(f"优化分块完成，共{len(chunks)}个块")

        # 9. 存入向量库
        for idx, chunk in enumerate(chunks):
            try:
                collection.add(
                    ids=[f"chunk_{idx}"],
                    documents=[chunk],
                    metadatas=[{
                        "source": pdf_path,
                        "chunk_id": idx,
                        "length": len(chunk)  # 记录块长度，便于后续分析
                    }]
                )
                if (idx + 1) % 5 == 0:  # 每5个块打印一次进度
                    print(f"已存入{idx+1}/{len(chunks)}个块")
            except Exception as e:
                print(f"存入第{idx}个块时出错：{str(e)}")
                continue

        print(f"全部完成，成功存入{len(chunks)}个块！")

    except Exception as e:
        print(f"程序运行出错：{str(e)}")

if __name__ == "__main__":
    main()
    