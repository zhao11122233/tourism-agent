from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, create_before_model_callback, report_prompt_switch, get_runtime_context


class ReactAgent:
    def __init__(self):
        # 加载系统提示词
        system_prompt = load_system_prompts()

        # 将 middleware 应用到所有工具（监控日志 + 上下文标记）
        wrapped_tools = [
            monitor_tool(rag_summarize),
            monitor_tool(get_weather),
            monitor_tool(get_user_location),
            monitor_tool(get_user_id),
            monitor_tool(get_current_month),
            monitor_tool(fetch_external_data),
            monitor_tool(fill_context_for_report),
        ]

        # 创建 before_model 日志回调
        before_model_callback = create_before_model_callback()

        # 创建PromptTemplate对象
        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["input", "chat_history", "tools", "tool_names", "agent_scratchpad"]
        )

        # 创建ReAct agent
        agent = create_react_agent(
            llm=chat_model,
            tools=wrapped_tools,
            prompt=prompt
        )

        # 创建AgentExecutor（注入 before_model 回调）
        self.agent = AgentExecutor(
            agent=agent,
            tools=wrapped_tools,
            callbacks=[before_model_callback],
            handle_parsing_errors=True,
            verbose=False,
        )

    def execute_stream(self, query: str):
        input_dict = {
            "input": query,
            "chat_history": ""
        }

        # 流式执行
        for chunk in self.agent.stream(input_dict):
            if "output" in chunk:
                yield chunk["output"] + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
