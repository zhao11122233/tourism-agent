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

# 加载.env文件中的环境变量
load_dotenv()

# 获取DASHSCOPE_API_KEY
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> BaseChatModel:
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            api_key=SecretStr(DASHSCOPE_API_KEY) if DASHSCOPE_API_KEY else None
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=DASHSCOPE_API_KEY
        )


chat_model = ChatModelFactory().generator()

embed_model = EmbeddingsFactory().generator()
