# 第十二章：旅游 Agent（Travel Agent）

## 一、本节学习目标

- 理解 LangGraph StateGraph 状态管理机制
- 掌握意图路由 + 信息收集 + 业务节点 的图结构设计
- 学会条件边（conditional edge）实现动态流转
- 理解两阶段政策咨询模式的设计

## 二、核心知识点讲解

### 2.1 LangGraph vs 传统 ReAct

| | ReAct Agent（第 11 章） | TravelAgent（LangGraph） |
|---|---|---|
| 控制流 | LLM 自由决定调哪个工具 | 预定义的节点 + 条件路由 |
| 状态管理 | 隐式（在 chat_history 里） | 显式 State（TypedDict） |
| 信息收集 | LLM 自行追问 | 结构化 collect_info 节点 |
| 可预测性 | 依赖 LLM 判断 | 代码逻辑明确 |
| 适用场景 | 简单问答 | 多步骤业务闭环 |

### 2.2 StateGraph 核心概念

```python
class TravelState(TypedDict):
    messages: list          # 对话历史
    scenic_spot: str        # 景区名
    visit_date: str         # 游玩日期
    traveler_count: int     # 人数
    phone: str              # 手机号
    intent: str             # 当前意图
    missing_fields: list    # 缺失的必填字段
    collect_step: int       # 信息收集步数
    final_answer: str       # 最终回答
```

- **State 是图的"内存"**：每个节点读写同一个 State
- **节点 = 函数**：接收 State，返回部分更新的 dict
- **边 = 流转方向**：普通边（固定）和条件边（动态）

### 2.3 节点职责

| 节点 | 职责 |
|------|------|
| `parse_intent` | 调 `parse_user_info` 解析意图 + 提取景区/日期 |
| `collect_info` | 检查必填字段，生成追问 |
| `ticket_inquiry` | 调 `ticket_query` 查余票 |
| `ticket_booking` | 调 `ticket_book` 订票 |
| `policy_query` | **两阶段模式**：概述收费方案 → 收集信息 → 精准计算 |
| `route_planning` | 调 `plan_route` 生成路线 |
| `narration` | 调 `guide_order_exec` 景点讲解 |
| `general` | 通用 LLM 回答（兜底） |

### 2.4 两阶段政策咨询模式

这是第 12 章的核心设计，将票价咨询分为两个阶段：

```
阶段 1（无人员信息）："故宫门票多少钱"
  → search_policy 检索景区收费方案
  → 展示价格 + 邀请："需要帮您精准计算吗？请告知人数、年龄、身份、日期"
  → 用户说"不用了" → 友好结束

阶段 2（有人员信息）："两个70岁老人6月15号去故宫"
  → 检查信息完整性（日期？年龄？）
  → 不完整 → 追问
  → 完整 → search_policy → verify_discount → calc_ticket_price
  → 展示：每人明细 + 附加项目提醒 + 总价 + AI 免责声明
```

### 2.5 条件边（Conditional Edge）

```python
def route_after_parse(state: TravelState) -> str:
    intent = state.get("intent", "general")
    missing = state.get("missing_fields", [])
    if missing:
        return "collect_info"
    return intent  # 直接走到对应的业务节点

workflow.add_conditional_edges(
    "parse_intent",
    route_after_parse,
    {
        "collect_info": "collect_info",
        "ticket_inquiry": "ticket_inquiry",
        "policy_query": "policy_query",
        "route_planning": "route_planning",
        "narration": "narration",
        "general": "general",
    }
)
```

`route_after_parse` 返回字符串 → 图框架根据映射字典走到对应节点。

## 三、项目落地场景

- `agent/travel_agent.py`：180+ 行构建完整 LangGraph 图
- `app.py` 调用 `TravelAgent().execute_stream(query, chat_history)`
- 信息提取 `_extract_from_text` 从用户输入自动识别景区、日期、人数、手机号
- 短追问上下文增强："故宫"跟在"老人有什么优惠"后面 → 自动拼成"老人去故宫有什么优惠"

## 四、关键代码+逐行注释

### 4.1 构建 Graph
```python
def build_travel_graph():
    workflow = StateGraph(TravelState)

    # 添加所有节点
    workflow.add_node("parse_intent", node_parse_intent)
    workflow.add_node("collect_info", node_collect_info)
    workflow.add_node("ticket_inquiry", node_ticket_inquiry)
    workflow.add_node("ticket_booking", node_ticket_booking)
    workflow.add_node("policy_query", node_policy_query)
    workflow.add_node("route_planning", node_route_planning)
    workflow.add_node("narration", node_narration)
    workflow.add_node("general", node_general)

    # 入口
    workflow.set_entry_point("parse_intent")

    # 意图解析后 → 条件路由
    workflow.add_conditional_edges("parse_intent", route_after_parse, {...})

    # 信息收集后 → 如果缺失字段则 END（返回追问）
    workflow.add_conditional_edges("collect_info", route_after_collect, {...})
    workflow.add_edge("collect_info", END)

    # 所有业务节点 → END
    for node in ["ticket_inquiry", "ticket_booking", "policy_query",
                 "route_planning", "narration", "general"]:
        workflow.add_edge(node, END)

    return workflow.compile()
```

### 4.2 信息收集节点
```python
def node_collect_info(state: TravelState) -> dict:
    intent = state.get("intent", "general")

    # 各意图的必填字段
    required_map = {
        "ticket_inquiry": ["scenic_spot", "visit_date"],
        "ticket_booking": ["scenic_spot", "visit_date", "phone"],
        "route_planning": ["scenic_spot"],
        "policy_query": [],    # ← 政策咨询阶段 1 不需要必填
        "narration": [],
        "general": [],
    }
    required = required_map.get(intent, [])

    # 检查缺失字段
    missing = [f for f in required if not state.get(f)]

    if not missing:
        return {"missing_fields": [], "collect_step": 0}

    # 一次只问一个字段
    prompts = {
        "scenic_spot": "请问您想去哪个景区呢？",
        "visit_date": "请问您计划哪天去呢？（如 2026-06-15）",
        "phone": "预订需要手机号，请告知您的 11 位手机号码。",
    }
    return {"missing_fields": missing, "final_answer": prompts[missing[0]]}
```

注意 `policy_query` 的 `required` 是空列表——阶段 1 不需要必填字段，用户只问"多少钱"就直接展示方案。

### 4.3 信息提取（_extract_from_text）
```python
def _extract_from_text(self, text: str):
    # 景区名 → 从预定义列表匹配
    spots = ["故宫", "八达岭长城", "黄山", "西湖", "兵马俑", ...]
    for s in spots:
        if s in text:
            self._state["scenic_spot"] = s

    # 日期 → 明天/后天/YYYY-MM-DD
    if "明天" in text:
        self._state["visit_date"] = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    # 人数 → "3个人"/"两位"
    count_match = re.search(r'(\d+)\s*[个位名]', text)
    if count_match:
        self._state["traveler_count"] = int(count_match.group(1))

    # 手机号 → 1 开头 11 位
    phone_match = re.search(r'1[3-9]\d{9}', text)
    if phone_match:
        self._state["phone"] = phone_match.group(0)
```

## 五、验收标准+踩坑避坑点

### 5.1 验收标准
- [ ] "故宫门票多少钱" → 返回景区收费方案 + 邀请精准计算（阶段 1）
- [ ] "两个70岁老人6月去故宫" → 追问缺少的日期（阶段 2，信息不全）
- [ ] "两个70岁老人6月15号去故宫" → 返回每人明细 + 总价 + 免责声明（阶段 2，完整）
- [ ] "不用了" → 友好结束，不再追问
- [ ] "帮我订故宫明天 3 个人 13812345678" → 走到 ticket_booking 节点

### 5.2 踩坑避坑点
1. **State 持久性**：`TravelAgent` 用 `self._state` 维护跨轮对话状态，每次 `execute_stream` 只更新不重置
2. **一次只问一个字段**：`collect_info` 每次只追问一个字段，避免信息过载
3. **policy_query 不设必填字段**：阶段 1 不需要任何必填信息，设了反而阻止概述展示
4. **隐式意图识别**：`parse_user_info` 中"人口+景区"组合默认归为定价咨询（即使没有"多少钱"关键词）
5. **Graph 编译一次**：`build_travel_graph()` 在 `__init__` 中调用一次，所有请求共享同一个编译好的图
