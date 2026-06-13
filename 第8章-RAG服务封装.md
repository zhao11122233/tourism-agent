# 第八章：RAG 服务封装（RAG Service）

## 一、本节学习目标

- 理解 RAG 服务层的职责：检索 + 总结生成
- 掌握 LangChain LCEL（LangChain Expression Language）链式调用
- 学会将检索结果格式化为 LLM 可理解的上下文
- 理解 PromptTemplate 动态变量的使用

## 二、核心知识点讲解

### 2.1 RAG 服务做了什么？

```
用户提问 "故宫门票多少钱"
    ↓
retriever.invoke(question)  →  Top-3 文档块
    ↓
格式化为 【参考资料1】【参考资料2】【参考资料3】
    ↓
送入 PromptTemplate → LLM 总结 → 返回自然语言回答
```

### 2.2 LCEL 链式调用

```python
chain = prompt_template | print_prompt | chat_model | StrOutputParser()
```

`|` 是 LangChain 的管道操作符：
1. `prompt_template`：把变量填入模板，生成完整 prompt
2. `print_prompt`：调试用，打印完整 prompt 到控制台
3. `chat_model`：调用 LLM 生成回答
4. `StrOutputParser()`：把 LLM 返回的 `AIMessage` 对象转成纯字符串

### 2.3 rag_summarize 提示词模板

`prompts/rag_summarize.txt` 定义了 LLM 如何"看资料回答"：
- 回答必须完全基于参考资料（禁止编造）
- 中文回答，热情专业简洁
- 严格围绕用户提问，不扩展不追问
- 纯文本输出，禁止 JSON/列表等结构

这保证了 RAG 的回答**可控、可信、不跑题**。

### 2.4 为什么用 RagSummarizeService 而不是直接调 retriever？

| 直接调 retriever | RagSummarizeService |
|---|---|
| 返回原始文档块 | 返回自然语言总结 |
| 用户/Agent 需自己筛选 | LLM 自动提炼关键信息 |
| 块之间可能有重复/矛盾 | LLM 自动合并去重 |

## 三、项目落地场景

- `agent_tools.py` 中的 `search_policy` 工具直接调用 `rag.rag_summarize(query)`
- `guide_order_exec` 导览模式中，无内置讲解词时回退到 RAG 检索
- `node_policy_query` 阶段 1 展示景区收费方案时，通过 RAG 检索知识库

## 四、关键代码+逐行注释

### 4.1 RagSummarizeService 完整源码
```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from rag.vector_store import VectorStoreService
from model.factory import chat_model
from utils.prompt_loader import load_rag_prompts

class RagSummarizeService:
    def __init__(self):
        # 1) 初始化向量库和检索器
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()  # k=3

        # 2) 加载 RAG 专用提示词模板
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)

        # 3) 获取对话模型（qwen-turbo）
        self.model = chat_model

        # 4) 构建 LCEL 链
        self.chain = self._init_chain()

    def _init_chain(self):
        # prompt_template → (调试打印) → LLM → 字符串输出
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        """检索原始文档块（供需要直接操作文档的场景使用）"""
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """核心方法：检索 + 总结，一步完成"""

        # 1) 检索 Top-3 文档块
        context_docs = self.retriever_docs(query)

        # 2) 格式化为 LLM 可理解的参考资料
        context = ""
        for i, doc in enumerate(context_docs, 1):
            context += (
                f"【参考资料{i}】: 参考资料：{doc.page_content}"
                f" | 参考元数据：{doc.metadata}\n"
            )

        # 3) 填入模板变量，调用 LLM 生成总结
        return self.chain.invoke({
            "input": query,      # 用户原始问题
            "context": context,  # 格式化后的参考资料
        })
```

### 4.2 模块级单例（agent_tools.py 中的用法）
```python
# agent/tools/agent_tools.py 第 9 行
from rag.rag_service import RagSummarizeService

rag = RagSummarizeService()  # 模块级单例，所有工具共享
```

### 4.3 RAG 提示词模板结构（prompts/rag_summarize.txt）
```
你是专注于"基于参考资料总结"的AI助手...
### 输入信息
1. 用户提问：{input}
2. 参考资料：{context}
### 严格遵守以下约束
1. 内容合规...
2. 事实准确：回答必须完全基于参考资料...
3. 语言要求：仅用中文回答，语气热情、专业、简洁...
4. 聚焦提问：严格围绕用户原始提问总结...
5. 格式要求：仅输出概括内容本身，纯文本...
```

`{input}` 和 `{context}` 是动态变量，调用时由 `chain.invoke({"input": ..., "context": ...})` 注入。

## 五、验收标准+踩坑避坑点

### 5.1 验收标准
- [ ] `rag.rag_summarize("故宫门票多少钱")` 返回包含价格数字的中文回答
- [ ] 回答内容来自知识库文件，不凭空编造
- [ ] 重复调用相同 query，回答基本一致（检索结果稳定）
- [ ] 知识库更新后，新内容能被检索到

### 5.2 踩坑避坑点
1. **全局单例是"冷启动"**：`rag = RagSummarizeService()` 在模块加载时执行，意味着 import 时就连接了 ChromaDB。如果 `chroma_db/` 不存在会报错，必须先 `python -m rag.vector_store` 构建向量库
2. **context 长度**：k=3 × chunk_size=200 = 最多 600 字符的参考资料，加上 prompt 模板和用户提问，总长度在 1K token 左右，qwen-turbo（8K 上下文）完全够用
3. **换知识库要重建**：修改 `data/knowledge/` 下的 .txt 文件后，需删除 `chroma_db/` 和 `md5.txt` 重建，否则检索到的还是旧内容
4. **retriever_docs vs rag_summarize**：前者返回原始文档（调试用），后者返回自然语言（生产用）
