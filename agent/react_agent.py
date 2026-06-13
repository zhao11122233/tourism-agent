from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (parse_user_info, search_policy, verify_discount, calc_ticket_price,
                                     plan_route, guide_order_exec, ticket_query, ticket_book)
from agent.tools.middleware import monitor_tool, create_before_model_callback, prompt_switch, get_runtime_context, reset_runtime_context


class ReactAgent:
    def __init__(self):
        # 加载系统提示词
        system_prompt = load_system_prompts()

        # 将 middleware 应用到所有工具（监控日志 + 上下文标记）
        wrapped_tools = [
            monitor_tool(parse_user_info),
            monitor_tool(search_policy),
            monitor_tool(verify_discount),
            monitor_tool(calc_ticket_price),
            monitor_tool(plan_route),
            monitor_tool(guide_order_exec),
            monitor_tool(ticket_query),
            monitor_tool(ticket_book),
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
            max_iterations=5,
        )

    def execute_stream(self, query: str, chat_history: str = ""):
        # 每次执行前重置运行时上下文
        reset_runtime_context()
        input_dict = {
            "input": query,
            "chat_history": chat_history
        }

        # 流式执行
        for chunk in self.agent.stream(input_dict):
            if "output" in chunk:
                yield chunk["output"] + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("我有2个老人和1个儿童，想去故宫游玩，帮我看看有什么优惠政策"):
        print(chunk, end="", flush=True)
