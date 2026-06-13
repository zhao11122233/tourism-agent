# 第七章：RAG 向量检索基础

## 一、本节学习目标

- 理解 RAG（检索增强生成）的工作原理
- 掌握 ChromaDB 向量数据库的基本使用
- 学会文档切分策略的设计与调参
- 理解 Embedding 向量的生成与相似度检索

## 二、核心知识点讲解

### 2.1 RAG 是什么？

```
用户提问 → 文本转向量 → 向量库检索 → Top-K 相关文档 → 拼接给 LLM → 生成回答
```

- **没有 RAG**：LLM 只能靠训练数据回答，不知道景区最新票价
- **有了 RAG**：LLM 从知识库检索到"故宫旺季 60 元"，回答基于真实数据
- 本质：给 LLM 配一个"外挂知识库"

### 2.2 ChromaDB 向量数据库

| 概念 | 说明 | 本项目取值 |
|------|------|------------|
| Collection | 向量集合（类似数据库的表） | `agent` |
| Embedding | 文本→数字向量（1024 维浮点数组） | `text-embedding-v4` |
| persist_directory | 向量持久化目录 | `chroma_db/` |
| k（Top-K） | 每次检索返回几个最相关文档块 | `3` |

### 2.3 文档切分（Chunking）

```python
RecursiveCharacterTextSplitter(
    chunk_size=200,        # 每块最多 200 字符
    chunk_overlap=20,      # 相邻块重叠 20 字符
    separators=["\n\n", "。", ".", "?", "？", "!", " ", ""]
)
```

切分顺序：先按双换行（段落）→ 句号 → 问号 → 感叹号 → 空格 → 逐字符。**越靠前的分隔符优先级越高**。

为什么 overlap=20？防止关键信息刚好卡在边界被切断。比如"旺季成人 60 元"如果恰好断在"旺季成人 "和"60 元"之间，LLM 就看不到完整价格。

### 2.4 Embedding 生成

- 文本 → `DashScopeEmbeddings` → 1024 维浮点向量
- 语义相近的文本 → 向量距离近 → 检索时排前面
- 本项目使用阿里云 `text-embedding-v4`，免费额度足够学习

## 三、项目落地场景

- `rag/vector_store.py`：文档加载 → 切分 → 转向量 → 存入 ChromaDB
- 命令行执行 `python -m rag.vector_store` 即可重建知识库
- MD5 去重机制：文件内容不变则不重复索引

## 四、关键代码+逐行注释

### 4.1 config/chroma.yml（向量库参数）
```yaml
collection_name: agent              # 集合名
persist_directory: chroma_db        # 持久化目录
k: 3                                # 检索返回 Top-3
data_path: data/knowledge           # 知识文件目录
md5_hex_store: md5.txt              # MD5 去重记录文件
allow_knowledge_file_type: ["txt", "pdf"]  # 允许的文件类型

chunk_size: 200                     # 分块大小（字符）
chunk_overlap: 20                   # 块间重叠
separators: ["\n\n", "。", ".", "?", "？", "!", " ", ""]
```

### 4.2 VectorStoreService 初始化
```python
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorStoreService:
    def __init__(self):
        # 1) 打开/创建 ChromaDB 集合
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],  # "agent"
            embedding_function=embed_model,                  # DashScope text-embedding-v4
            persist_directory=chroma_conf["persist_directory"],  # "chroma_db"
        )

        # 2) 初始化文档切分器
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],        # 200
            chunk_overlap=chroma_conf["chunk_overlap"],  # 20
            separators=chroma_conf["separators"],
        )
```

### 4.3 文档加载流程（load_document）
```python
def load_document(self):
    # 1) 列出 data/knowledge/ 下所有 .txt 和 .pdf 文件
    allowed_files = listdir_with_allowed_type(
        get_abs_path("data/knowledge"),
        ("txt", "pdf"),
    )

    for file_path in allowed_files:
        # 2) MD5 去重：计算文件哈希，已处理过的跳过
        md5 = get_file_md5_hex(file_path)
        if check_md5_hex(md5):
            continue  # 内容未变，跳过

        # 3) 读取文件内容 → Document 对象
        documents = txt_loader(file_path)  # 或 pdf_loader

        # 4) 切分 Document
        chunks = self.spliter.split_documents(documents)

        # 5) 分批存入 ChromaDB（DashScope API 限制每次最多 10 条）
        for i in range(0, len(chunks), 10):
            batch = chunks[i:i + 10]
            self.vector_store.add_documents(batch)

        # 6) 记录 MD5，下次跳过
        save_md5_hex(md5)
```

### 4.4 检索器
```python
def get_retriever(self):
    return self.vector_store.as_retriever(
        search_kwargs={"k": 3}  # 每次返回最相关的 3 个文档块
    )

# 使用示例
retriever = vs.get_retriever()
docs = retriever.invoke("故宫门票多少钱")
for doc in docs:
    print(doc.page_content)  # 最相关的 3 段文本
```

## 五、验收标准+踩坑避坑点

### 5.1 验收标准
- [ ] `python -m rag.vector_store` 执行无报错，日志显示"内容加载成功"
- [ ] `chroma_db/` 目录生成，包含向量库文件
- [ ] `retriever.invoke("故宫")` 返回 3 条相关文档
- [ ] 重复执行 `load_document()` 不会重复索引同一文件（MD5 去重生效）

### 5.2 踩坑避坑点
1. **切换 Embedding 模型必须清库**：不同模型的向量维度不同（v4=1024），换模型后 `chroma_db/` 必须删除重建，否则维度不匹配报错
2. **chunk_size 不是越小越好**：200 是本项目实测最优值。太小导致信息碎片化，太大导致检索精度下降
3. **DashScope API 批次限制**：`add_documents` 每次最多 10 条，必须分批提交
4. **MD5 去重是增量而非全量**：旧文件的旧块不会自动删除，要完全重建需手动删除 `chroma_db/` 和 `md5.txt`
5. **中文分隔符**：separators 必须包含 `"。"`、`"？"`、`"！"`，否则中文文本会在奇怪的位置切断
