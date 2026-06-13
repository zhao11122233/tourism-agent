# 第十一章：ReAct Agent

## 一、本节学习目标

- 理解 ReAct（Reasoning + Acting）推理模式
- 掌握 LangChain `create_react_agent` + `AgentExecutor` 的使用
- 理解 Thought → Action → Observation 循环
- 学会工具监控装饰器和回调机制

## 二、核心知识点讲解

### 2.1 ReAct 模式

```
Question: 两个老人一个小孩去故宫有什么优惠？

Thought: 需要先解析用户的人员构成和意图
Action: parse_user_info
Action Input: {"query": "两个老人一个小孩去故宫有什么优惠？"}
Observation: {"person_types": ["老人","儿童"], "extracted_ages": [], ...}

Thought: 用户有老人和儿童，需要查故宫的优惠政策
Action: search_policy
Action Input: {"query": "故宫 老人 儿童 优惠政策"}
Observation: 故宫旺季成人60元，老人60-69半价，儿童6-18半价...

Thought: 已经知道优惠政策，现在判定优惠资格
Action: verify_discount
Action Input: {"person_types": "[\"老人\",\"儿童\"]", "certificates": "{}"}
Observation: [{"discount_type": "半价", "discount_rate": 0.5}, ...]

Thought: 优惠资格已判定，现在计算总价
Action: calc_ticket_price
Action Input: {"scenic_spot": "故宫", "person_details": "[...]", ...}
Observation: {"total_price": 90.0, ...}

Thought: I now know the final answer
Final Answer: 故宫旺季成人60元，您的两位老人各30元（半价），一位儿童30元（半价），合计90元...
```

每一步：**思考 → 行动 → 观察 → 再思考**，直到能给出最终答案。

### 2.2 两个核心组件

| 组件 | 作用 |
|------|------|
| `create_react_agent` | 把 LLM + 工具 + prompt 绑定成 ReAct Agent |
| `AgentExecutor` | 执行 Agent 循环：调工具、处理异常、限制迭代次数 |

### 2.3 AgentExecutor 参数

```python
AgentExecutor(
    agent=agent,
    tools=wrapped_tools,
    callbacks=[before_model_callback],  # 每次调 LLM 前记录日志
    handle_parsing_errors=True,         # LLM 输出格式错误时自动重试
    verbose=False,                      # 不打印完整执行过程到控制台
    max_iterations=5,                   # 最多执行 5 轮 Thought→Action
)
```

`max_iterations=5` 是安全阀：防止 Agent 陷入死循环无限调工具。

### 2.4 工具监控装饰器

```python
def monitor_tool(tool_func):
    """在工具调用前后记录日志"""
    @functools.wraps(original_func)
    def wrapped_func(*args, **kwargs):
        logger.info(f"[tool monitor] 执行工具: {tool_name}")
        logger.info(f"[tool monitor] 传入参数: {args}, {kwargs}")
        result = original_func(*args, **kwargs)
        logger.info(f"[tool monitor] 工具 {tool_name} 调用成功")
        return result
    tool_func.func = wrapped_func
    return tool_func
```

不修改工具逻辑，只在外层加日志——这是装饰器模式的经典用法。

### 2.5 流式输出

```python
for chunk in self.agent.stream(input_dict):
    if "output" in chunk:
        yield chunk["output"] + "\n"
```

AgentExecutor 的 `stream()` 方法返回生成器，每步执行完 yield 一个 chunk：
- `chunk` 可能包含 `"actions"`（正在调工具）、`"steps"`（调完）、`"output"`（最终答案）
- 只 yield `"output"` → Streamlit 前端逐字显示最终答案

## 三、项目落地场景

- `agent/react_agent.py`：ReactAgent 类封装完整 ReAct 流程
- `app.py` 调用 `agent.execute_stream(query, chat_history)` 获取流式回答
- 实际生产中使用的是 `TravelAgent`（LangGraph），ReactAgent 作为备选保留

## 四、关键代码+逐行注释

### 4.1 ReactAgent 完整源码
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (parse_user_info, search_policy, ...)
from agent.tools.middleware import monitor_tool, create_before_model_callback, reset_runtime_context

class ReactAgent:
    def __init__(self):
        # 1) 加载 system prompt（main_prompt.txt）
        system_prompt = load_system_prompts()

        # 2) 给每个工具套上监控装饰器
        wrapped_tools = [
            monitor_tool(parse_user_info),
            monitor_tool(search_policy),
            monitor_tool(verify_discount),
            monitor_tool(calc_ticket_price),
            monitor_tool(plan_route),
            monitor_tool(guide_order_exec),
        ]

        # 3) 创建 before_model 回调（调 LLM 前记日志）
        before_model_callback = create_before_model_callback()

        # 4) PromptTemplate：{tools} {tool_names} {input} {agent_scratchpad} 等变量
        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["input", "chat_history", "tools",
                             "tool_names", "agent_scratchpad"]
        )

        # 5) 创建 ReAct Agent
        agent = create_react_agent(
            llm=chat_model,       # qwen-turbo
            tools=wrapped_tools,
            prompt=prompt
        )

        # 6) 创建 AgentExecutor
        self.agent = AgentExecutor(
            agent=agent,
            tools=wrapped_tools,
            callbacks=[before_model_callback],
            handle_parsing_errors=True,
            verbose=False,
            max_iterations=5,
        )
```

### 4.2 流式执行
```python
def execute_stream(self, query: str, chat_history: str = ""):
    reset_runtime_context()  # 每次对话前清空上下文标记

    input_dict = {
        "input": query,
        "chat_history": chat_history
    }

    for chunk in self.agent.stream(input_dict):
        if "output" in chunk:
            yield chunk["output"] + "\n"
```

## 五、验收标准+踩坑避坑点

### 5.1 验收标准
- [ ] `ReactAgent().execute_stream("两个老人去故宫")` 返回包含优惠和价格的回答
- [ ] 工具调用日志正常输出（monitor_tool 生效）
- [ ] `max_iterations=5` 限制生效，不会无限循环
- [ ] LLM 输出格式错误时 `handle_parsing_errors=True` 自动重试

### 5.2 踩坑避坑点
1. **ReactAgent vs TravelAgent**：本项目主 Agent 是 `TravelAgent`（LangGraph），`ReactAgent` 是保留的备选。两者对外接口兼容（都有 `execute_stream`），可以互换
2. **`agent_scratchpad` 变量**：Prompt 模板中的 `{agent_scratchpad}` 由 LangChain 自动填充（历史 Thought/Action/Observation），不能删
3. **装饰器执行顺序**：`monitor_tool(parse_user_info)` 先于 AgentExecutor 执行，所以工具调用日志在 Agent 日志之前
4. **不要设置 verbose=True**：会把每次 Thought/Action/Observation 打印到 stderr，Streamlit 界面会被刷屏
