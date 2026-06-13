# 第五章：Prompt 工程（Prompts）

## 一、本节学习目标

- 理解 Prompt 在 LLM 应用中的核心地位
- 掌握 4 个核心 Prompt 模板的用途
- 学会通过 YAML + 文件分离的方式管理 Prompt
- 理解 ReAct 格式与角色约束的设计思路

## 二、核心知识点讲解

### 2.1 为什么 Prompt 单独成文件？
- **业务方友好**：产品/运营可直接修改，无需懂代码
- **版本管理**：用 git diff 就能看到 Prompt 变更
- **长 Prompt 不污染代码**：上千字的 prompt 写在代码里非常难维护
- **路径可配置**：换 Prompt 不用改代码，只改 `prompts.yml`

### 2.2 本项目 4 个 Prompt 矩阵
| 文件 | 角色 | 关键变量 |
|------|------|----------|
| `main_prompt.txt` | Agent 调度大脑 | `{tools}` `{tool_names}` `{input}` `{agent_scratchpad}` |
| `rag_summarize.txt` | RAG 答案总结员 | `{input}` `{context}` |
| `report_prompt.txt` | 健康报告写手 | （使用工具） |
| `order_output_prompt.txt` | 订单生成器 | `{context}` |

### 2.3 ReAct Prompt 经典结构
```
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```
**本质**：强制 LLM 按"思考→行动→观察"循环输出，Agent 才能解析。

## 三、项目落地场景

- 用户发问 → `main_prompt` 引导 LLM 决定调哪个工具
- RAG 检索回 Top-K 文档 → `rag_summarize` 把文档"翻译"成自然语言回答
- 订单确认 → `order_output` 把票务信息"渲染"成结构化 JSON
- 健康报告 → `report_prompt` 调度 get_user_id、fetch_external_data 等工具

## 四、关键代码+逐行注释

### 4.1 main_prompt.txt（Agent 主提示词）
```
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}

你是文旅智能助手，具备自主的ReAct思考与工具调用能力，严格遵循「思考→行动→观察→再思考」的流程为游客提供全链路服务，涵盖需求解析、优惠政策检索、优惠资格判定、票价计算、路线规划、导览讲解、查票订票。语气热情友好、专业可靠，让游客感受到贴心服务。

【上下文记忆铁律】当用户只说了一个短词（如"故宫""长城""3个人"），绝对不要直接当成导览请求或新问题！先查 Chat History 中用户之前的完整意图是什么。
```

### 4.2 rag_summarize.txt（RAG 总结提示词）
```
你是专注于"基于参考资料总结"的AI助手，擅长文旅票务、优惠政策、景区导览领域，需结合用户提问和向量检索到的参考资料，生成简洁准确的概括回答。

### 输入信息
1. 用户提问：{input}
2. 参考资料(在下一个###之前内容均为参考资料)：{context}

### 严格遵守以下约束（违反将导致回答无效）
1. 内容合规：禁止包含违法、侵权、攻击性信息
2. 事实准确：回答必须完全基于参考资料中的信息，不编造、不添加未提及的内容
3. 语言要求：仅用中文回答，语气热情、专业、简洁
4. 聚焦提问：严格围绕用户原始提问总结，不扩充问题范围
5. 格式要求：仅输出概括内容本身，以纯文本字符串形式呈现
```

### 4.3 order_output_prompt.txt（订单生成）
```
你是专业的文旅订单生成助手，负责根据已确认的票务信息和人员信息生成结构化预订单。

### 输入信息
已确认的票务上下文：{context}

### 输出要求
1. 根据上下文中的票务信息，生成一份完整、规范的结构化JSON预订单
2. JSON必须包含以下字段：
   - order_id：预订单ID（格式：PRE + 年月日 + 6位流水号）
   - platform：购票平台（meituan/ctrip/spot_self）
   - scenic_spot：景区名称
   - tickets：票务明细数组
   - total_price：总价（元）
   - platform_service_fee：平台服务费（元）
```

### 4.4 prompts.yml（路径映射）
```yaml
main_prompt_path: prompts/main_prompt.txt
rag_summarize_prompt_path: prompts/rag_summarize.txt
report_prompt_path: prompts/report_prompt.txt
order_output_prompt_path: prompts/order_output_prompt.txt
```

### 4.5 在 LangChain 中使用 Prompt
```python
from langchain_core.prompts import PromptTemplate
from utils.prompt_loader import load_system_prompts

# 加载原始字符串
raw = load_system_prompts()

# 包成 PromptTemplate（LangChain 标准对象）
prompt = PromptTemplate.from_template(raw)

# 用变量渲染
rendered = prompt.format(
    tools="ticket_query, ticket_book, plan_route",
    tool_names="ticket_query, ticket_book, plan_route",
    input="帮我订故宫明天的票，3个人",
    agent_scratchpad=""
)
print(rendered[:200])  # 打印前 200 字符
```

## 五、验收标准+踩坑避坑点

### 5.1 验收标准
- [ ] 4 个 Prompt 文件均能用 `load_xxx_prompts()` 成功读取
- [ ] `PromptTemplate.from_template` 渲染不报错
- [ ] 修改 `main_prompt.txt` 后 Agent 行为有可见变化
- [ ] 故意把 `prompts.yml` 中某路径改错，日志能定位到具体 Prompt

### 5.2 踩坑避坑点
1. **占位符大小写敏感**：`{input}` 和 `{Input}` 是不同的，LangChain 不会自动归一化
2. **JSON 大括号冲突**：`order_output_prompt.txt` 中有大量 `{}` 示例，**用 `PromptTemplate.from_template` 可能会误解析**——这种纯变量替换的场景直接 `str.format` 即可
3. **Prompt 长度限制**：qwen-turbo 上下文 8K，超长会截断；用 `qwen-plus`（32K）或精简 Prompt
4. **中文标点**：Prompt 中务必用全角中文标点，避免 LLM 误读
5. **多轮对话慎用 `PromptTemplate`**：多轮场景要改用 `ChatPromptTemplate` + `MessagesPlaceholder`
