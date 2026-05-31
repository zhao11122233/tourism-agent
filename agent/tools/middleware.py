import functools
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain_core.callbacks import BaseCallbackHandler

# 模块级别的运行时上下文，用于在工具调用间传递状态
_runtime_context: dict = {}


def get_runtime_context() -> dict:
    """获取运行时上下文"""
    return _runtime_context


def monitor_tool(tool_func):
    """
    工具调用监控装饰器：
    - 记录工具名称和参数
    - 记录调用成功/失败
    - 对 fill_context_for_report 设置 report 上下文标记
    """
    tool_name = tool_func.name
    original_func = tool_func.func

    @functools.wraps(original_func)
    def wrapped_func(*args, **kwargs):
        logger.info(f"[tool monitor]执行工具:{tool_name}")
        logger.info(f"[tool monitor]传入参数: args={args}, kwargs={kwargs}")

        try:
            result = original_func(*args, **kwargs)
            logger.info(f"[tool monitor]工具{tool_name}调用成功")

            if tool_name == "fill_context_for_report":
                _runtime_context["report"] = True

            return result
        except Exception as e:
            logger.error(f"[tool monitor]工具{tool_name}调用失败:{e}")
            raise e

    # 将包装后的函数放回工具对象
    tool_func.func = wrapped_func
    return tool_func


class BeforeModelCallback(BaseCallbackHandler):
    """
    模型调用前日志回调：
    - 记录消息数量
    - 记录最后一条消息的类型和内容摘要
    """

    def on_chat_model_start(self, serialized, messages, **kwargs):
        logger.info(f"[log_before_model]即将调用模型，带有{len(messages)}条消息")
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, (list, tuple)):
                last_msg = last_msg[0] if last_msg else None
            if last_msg:
                content = getattr(last_msg, 'content', str(last_msg))
                logger.debug(f"[log_before_model]{type(last_msg).__name__} {str(content)[:200]}")
        return None


def create_before_model_callback() -> BeforeModelCallback:
    """创建 before_model 回调实例"""
    return BeforeModelCallback()


def report_prompt_switch() -> str:
    """
    动态提示词切换：
    根据运行时上下文中是否有 report 标记，返回对应的系统提示词
    """
    is_report = _runtime_context.get("report", False)
    if is_report:
        return load_report_prompts()
    else:
        return load_system_prompts()


def reset_runtime_context():
    """重置运行时上下文"""
    _runtime_context.clear()
