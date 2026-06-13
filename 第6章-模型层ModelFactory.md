# 第六章：模型层（Model Factory）

## 一、本节学习目标

- 理解"模型工厂"模式的设计价值
- 掌握阿里云百炼（DashScope）的对接方式
- 能够封装 Chat 模型与 Embedding 模型
- 了解 BaseModel 抽象类的工厂方法

## 二、核心知识点讲解

### 2.1 什么是模型工厂？
- 避免在业务代码里到处 `ChatTongyi(...)` 写死
- 通过配置切换 `qwen-turbo` / `qwen-plus` / `qwen-max`
- 统一管理 API Key、超时、重试等参数
- 后续接 DeepSeek、OpenAI 时只改工厂，不改业务

### 2.2 为什么用抽象基类（ABC）？
```python
class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> ...: pass
```
- 强制子类实现 `generator()` 方法
- 业务代码只依赖 `BaseModelFactory` 抽象，不依赖具体厂商
- 单元测试时可以传入 `MockModelFactory` 替换

### 2.3 阿里云百炼两大模型
| 类别 | 类 | 本项目使用 | 作用 |
|------|----|------------|------|
| Chat | `ChatTongyi` | `qwen-turbo` | 对话生成、Agent 推理 |
| Embedding | `DashScopeEmbeddings` | `text-embedding-v4` | 文本向量化、RAG |

### 2.4 API Key 管理
- **不写在代码里**（防止泄露）
- 通过 `.env` 文件 + `python-dotenv` 加载
- `.env` 加入 `.gitignore`

## 三、项目落地场景

- Agent 层：调用 `chat_model.invoke(messages)` 让 LLM 推理
- RAG 层：调用 `embed_model.embed_documents([...])` 把文档转向量
- 切换模型：只改 `config/rag.yml` 一行，重启即可

## 四、关键代码+逐行注释

### 4.1 .env 模板
```bash
# 阿里云百炼 API Key
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 4.2 model/factory.py（完整源码）
```python
from abc import ABC, abstractmethod
from typing import Optional
import os
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from pydantic import SecretStr
from utils.config_handler import rag_conf

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取 DASHSCOPE_API_KEY
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


class BaseModelFactory(ABC):
    """模型工厂抽象基类"""
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    """对话模型工厂"""
    def generator(self) -> BaseChatModel:
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            api_key=SecretStr(DASHSCOPE_API_KEY) if DASHSCOPE_API_KEY else None
        )


class EmbeddingsFactory(BaseModelFactory):
    """Embedding 模型工厂"""
    def generator(self) -> Embeddings:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=DASHSCOPE_API_KEY
        )


# 模块级单例：项目里直接 import 即可使用
chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
```

### 4.3 最小调用测试
```python
from model.factory import chat_model, embed_model

# 1) Chat 模型
response = chat_model.invoke("你好，请用一句话介绍故宫")
print(response.content)

# 2) Embedding 模型
vectors = embed_model.embed_documents(["故宫", "长城", "西湖"])
print(len(vectors), len(vectors[0]))  # 3 1024（v4 默认 1024 维）
```

### 4.4 config/rag.yml
```yaml
chat_model_name: qwen-turbo
embedding_model_name: text-embedding-v4
```

### 4.5 进阶：扩展多模型切换
```python
class ChatModelFactory(BaseModelFactory):
    def generator(self) -> BaseChatModel:
        model_name = rag_conf["chat_model_name"]

        if model_name.startswith("qwen"):
            return ChatTongyi(
                model=model_name,
                api_key=SecretStr(DASHSCOPE_API_KEY) if DASHSCOPE_API_KEY else None
            )
        elif model_name.startswith("deepseek"):
            # 假设封装了 DeepSeekChat 类
            return DeepSeekChat(model=model_name, api_key=os.getenv("DEEPSEEK_API_KEY"))
        else:
            raise ValueError(f"不支持的模型: {model_name}")
```

## 五、验收标准+踩坑避坑点

### 5.1 验收标准
- [ ] `.env` 文件配置 `DASHSCOPE_API_KEY` 后，`os.getenv` 能取到
- [ ] `chat_model.invoke("hi")` 能返回非空字符串
- [ ] `embed_model.embed_documents(["a"])` 返回 `[[float, float, ...]]`
- [ ] 修改 `rag.yml` 中 `chat_model_name` 为 `qwen-plus` 后立即生效
- [ ] 不设置 `DASHSCOPE_API_KEY` 时程序能给出明确报错

### 5.2 踩坑避坑点
1. **`SecretStr` 包装**：Pydantic v2 之后，API Key 必须用 `SecretStr` 包装，否则会有安全警告
2. **`load_dotenv` 位置**：必须在 `os.getenv` 之前调用；建议放在模块顶部
3. **Embedding 维度一致**：切换 Embedding 模型时，**必须清空 `chroma_db` 目录**，否则维度不匹配会报错
4. **模型上下文长度**：
   - `qwen-turbo`：8K
   - `qwen-plus`：32K
   - `qwen-max`：32K
5. **网络代理**：DashScope SDK 默认走 `dashscope.aliyuncs.com`，公司内网可能需要配代理
6. **`.env` 千万别提交**：`.gitignore` 必须有 `.env`（本项目已配置）
