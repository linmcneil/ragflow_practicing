from PyPDF2 import PdfReader
import chromadb
from chromadb.utils import embedding_functions
import os
import re
import json
from modelscope import snapshot_download

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def main():
    try:
        # 1. 下载中文嵌入模型
        embedding_model_dir = os.path.join(os.getcwd(), "models", "bge-small-zh-v1.5")
        if not os.path.exists(embedding_model_dir):
            print("正在下载中文嵌入模型...")
            snapshot_download(
                model_id='Xorbits/bge-small-zh-v1.5',
                local_dir=embedding_model_dir
            )

        # 2. 检查数据集文件
        dataset_path = "train.json"
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"成功加载数据集，共{len(dataset)}条记录")
        except FileNotFoundError:
            print(f"错误：未找到数据集文件 '{dataset_path}'")
            return
        except json.JSONDecodeError:
            print(f"错误：数据集文件 '{dataset_path}' 不是有效的JSON格式")
            return

        # 3. 初始化Chroma客户端
        chroma_client = chromadb.PersistentClient(path="./chroma_data")
        try:
            chroma_client.delete_collection(name="knowledges")
        except:
            pass

        # 4. 初始化嵌入函数
        try:
            embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_dir,
                device="cuda"
            )
            print("嵌入函数初始化成功")
        except Exception as e:
            print(f"嵌入函数初始化失败：{str(e)}")
            return

        # 5. 创建集合
        try:
            collection = chroma_client.create_collection(
                name="knowledges",
                embedding_function=embedding_func
            )
            print("集合创建成功")
        except Exception as e:
            print(f"集合创建失败：{str(e)}")
            return

        # 6. 从数据集中提取文本（适配instruction-input-output格式）
        texts = []
        for idx, item in enumerate(dataset):
            # 提取核心字段
            instruction = item.get("instruction", "").strip()
            input_text = item.get("input", "").strip()
            output = item.get("output", "").strip()
            history = item.get("history", "")
            
            # 格式化历史对话（如果有）
            history_text = ""
            if history and isinstance(history, list):
                history_parts = []
                for h in history:
                    if isinstance(h, (list, tuple)) and len(h) >= 2:
                        history_parts.append(f"问：{h[0]}\n答：{h[1]}")
                if history_parts:
                    history_text = "\n【历史对话】".join(history_parts)
            
            # 组合成完整文本块（保留字段结构便于检索）
            parts = []
            if instruction:
                parts.append(f"【问题】{instruction}")
            if input_text:
                parts.append(f"【详情】{input_text}")
            if history_text:
                parts.append(f"【历史】{history_text}")
            if output:
                parts.append(f"【解答】{output}")
            
            full_text = "\n".join(parts).strip()
            if full_text:
                texts.append(full_text)

        if not texts:
            print("错误：未从数据集中提取到任何文本")
            return

        # 7. 优化分块策略（针对问答格式专项优化）
        chunks = []
        max_chunk_size = 150  # 问答文本适合更小的块
        hard_limit = 250       # 强制拆分上限
        
        for text in texts:
            text_clean = re.sub(r'\s+', ' ', text).strip()
            if not text_clean:
                continue
            
            current_position = 0
            text_length = len(text_clean)
            
            while current_position < text_length:
                remaining_length = text_length - current_position
                
                if remaining_length <= max_chunk_size:
                    chunk = text_clean[current_position:].strip()
                    if chunk:
                        chunks.append(chunk)
                    break
                
                # 优先按字段分割（保留【问题】【解答】等完整字段）
                field_match = re.search(r'【[^】]+】', text_clean[current_position + max_chunk_size:current_position + hard_limit])
                if field_match:
                    # 在新字段开始前分割
                    chunk_end = current_position + max_chunk_size + field_match.start()
                    chunk = text_clean[current_position:chunk_end].strip()
                    current_position = chunk_end
                else:
                    # 按句尾分割
                    end_pos = current_position + max_chunk_size
                    search_end = min(current_position + hard_limit, text_length)
                    match = re.search(r'[。！？；]', text_clean[end_pos:search_end])
                    
                    if match:
                        chunk_end = end_pos + match.end()
                        chunk = text_clean[current_position:chunk_end].strip()
                        current_position = chunk_end
                    else:
                        chunk = text_clean[current_position:current_position + max_chunk_size].strip()
                        current_position += max_chunk_size
                
                if chunk:
                    chunks.append(chunk)

        # 去重但保留顺序
        chunks = list(dict.fromkeys(chunks))
        print(f"优化分块完成，共{len(chunks)}个块")

        # 8. 存入向量库
        for idx, chunk in enumerate(chunks):
            try:
                collection.add(
                    ids=[f"chunk_{idx}"],
                    documents=[chunk],
                    metadatas=[{
                        "source": dataset_path,
                        "chunk_id": idx,
                        "length": len(chunk),
                        "contains_instruction": "【问题】" in chunk,
                        "contains_answer": "【解答】" in chunk
                    }]
                )
                if (idx + 1) % 10 == 0:
                    print(f"已存入{idx+1}/{len(chunks)}个块")
            except Exception as e:
                print(f"存入第{idx}个块时出错：{str(e)}")
                continue

        print(f"全部完成，成功存入{len(chunks)}个块！")

    except Exception as e:
        print(f"程序运行出错：{str(e)}")

if __name__ == "__main__":
    main()
