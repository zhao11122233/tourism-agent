# 第三章：工具函数层（Utils）

## 一、本节学习目标

- 理解工具函数层的定位与价值
- 掌握日志模块的封装方法
- 掌握文件 MD5 计算的用途
- 掌握 Prompt 文件的加载机制
- 能够独立编写带异常处理的文件处理工具

## 二、核心知识点讲解

### 2.1 Utils 层定位
`utils/` 是项目的**通用工具箱**，提供所有模块都会复用的能力：
- `path_tool.py` → 路径管理
- `logger_handler.py` → 日志系统
- `file_handler.py` → 文件操作、MD5
- `prompt_loader.py` → Prompt 加载
- `config_handler.py` → 配置加载（已在第二章讲过）

**设计原则**：纯函数、零业务依赖、可独立测试。

### 2.2 日志体系设计
为什么要封装日志？
- 统一格式（时间、模块、级别、文件:行号）
- 同时输出到控制台和文件
- 避免重复添加 Handler 导致日志翻倍
- 控制不同级别（控制台 INFO，文件 DEBUG）

### 2.3 MD5 的作用
- 用于判断知识库文档**是否被修改**
- Chroma 入库时记录 `md5.txt`，下次启动跳过未变化的文件
- 增量更新、向量化复用，避免重复计算 Embedding

### 2.4 Prompt 加载机制
- Prompt 不写在代码里（不利于维护）
- 单独存为 `.txt` 文件
- 通过 `prompts.yml` 配置路径
- 启动时一次性加载到内存

## 三、项目落地场景

- 程序每次运行都会用到 `logger` 输出运行状态
- RAG 模块会用 `get_file_md5_hex` 判断文档变化
- Agent 会通过 `load_system_prompts()` 获取系统提示词
- 所有模块都通过 `get_abs_path` 解决"路径不一致"问题

## 四、关键代码+逐行注释

### 4.1 日志封装 utils/logger_handler.py
```python
import logging
from utils.path_tool import get_abs_path
import os
from datetime import datetime

# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")
# 确保日志目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的统一格式
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

def get_logger(
    name: str = "agent",
    console_level: int = logging.INFO,    # 控制台只显示 INFO 以上
    file_level: int = logging.DEBUG,     # 文件记录 DEBUG 全量
    log_file=None,
) -> logging.Logger:
    """获取一个配置好的日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 Handler（Streamlit 重载时常见）
    if logger.handlers:
        return logger

    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    # 文件 Handler（按天滚动）
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger

# 模块级单例：项目里直接 import logger 即可使用
logger = get_logger()

if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("调试日志")
```

### 4.2 文件处理 utils/file_handler.py
```python
import os
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def get_file_md5_hex(filepath: str):
    """获取文件的 MD5 十六进制字符串"""
    if not os.path.exists(filepath):
        logger.error(f"[MD5计算]文件{filepath}不存在")
        return
    if not os.path.isfile(filepath):
        logger.error(f"[MD5计算]路径{filepath}不是文件")
        return

    md5_obj = hashlib.md5()
    chunk_size = 4096   # 4KB 分片，避免大文件爆内存
    try:
        with open(filepath, "rb") as f:    # 必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
        return md5_obj.hexdigest()
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None

def listdir_with_allowed_type(path: str, allowed_types: tuple):
    """返回文件夹内指定后缀的文件列表"""
    files = []
    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return tuple()

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))
    return tuple(files)

def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    """PDF 加载器封装"""
    return PyPDFLoader(filepath, passwd).load()

def txt_loader(filepath: str) -> list[Document]:
    """TXT 加载器封装"""
    return TextLoader(filepath, encoding="utf-8").load()
```

### 4.3 Prompt 加载器 utils/prompt_loader.py
```python
from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def load_system_prompts():
    """加载 Agent 主提示词"""
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]yaml 中没有 main_prompt_path 配置项")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e

def load_rag_prompts():
    """加载 RAG 总结提示词"""
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]yaml 中没有 rag_summarize_prompt_path 配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析 RAG 提示词出错，{str(e)}")
        raise e

def load_report_prompts():
    """加载报告生成提示词"""
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]yaml 中没有 report_prompt_path 配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告生成提示词出错，{str(e)}")
        raise e

def load_order_output_prompts():
    """加载订单输出提示词"""
    try:
        order_prompt_path = get_abs_path(prompts_conf["order_output_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_order_output_prompts]yaml 中没有 order_output_prompt_path 配置项")
        raise e

    try:
        return open(order_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_order_output_prompts]解析订单输出提示词出错，{str(e)}")
        raise e
```

## 五、验收标准+踩坑避坑点

### 5.1 验收标准
- [ ] 运行 `python utils/logger_handler.py` 能在控制台和 `logs/` 目录看到日志
- [ ] `get_file_md5_hex("data/knowledge/票务价格规则.txt")` 返回 32 位十六进制
- [ ] `load_system_prompts()` 返回 `prompts/main_prompt.txt` 的全部文本
- [ ] 故意改坏 `prompts.yml` 中的路径，日志能输出错误

### 5.2 踩坑避坑点
1. **Streamlit 日志重复**：Streamlit 会在文件变更时重新执行模块，导致 logger 被多次实例化。**`if logger.handlers: return logger`** 这一行就是用来防御的
2. **MD5 必须二进制读**：用 `"r"` 文本模式读会因编码差异导致 MD5 不一致
3. **大文件读法**：不要 `f.read()` 一次性读，用 4KB 分片循环
4. **Prompt 编码**：所有 .txt 文件统一用 UTF-8，否则 Windows 默认 GBK 读取会报错
5. **`utils/__init__.py` 必须存在**：哪怕是空文件，否则 `from utils.xxx` 会失败
