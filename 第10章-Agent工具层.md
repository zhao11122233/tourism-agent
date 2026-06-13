# 第十章：Agent 工具层（Tools）

## 一、本节学习目标

- 掌握 LangChain `@tool` 装饰器的用法
- 理解 8 个文旅工具的职责边界和调用顺序
- 学会工具参数校验和容错设计
- 掌握装饰器模式实现工具监控

## 二、核心知识点讲解

### 2.1 `@tool` 装饰器

```python
from langchain_core.tools import tool

@tool
def parse_user_info(query: str) -> str:
    """解析用户原始查询..."""
    # 函数实现
    return json.dumps(result, ensure_ascii=False)
```

`@tool` 把普通 Python 函数变成 LangChain Tool：
- 函数名 → 工具名
- docstring → 工具描述（**LLM 据此判断何时调用**）
- 参数签名 → 工具的输入 schema
- 返回值 → 工具的 Observation

### 2.2 8 个工具职责矩阵

| # | 工具 | 输入 | 输出 | 调用时机 |
|---|------|------|------|----------|
| 1 | `parse_user_info` | 用户原始文本 | 结构化 JSON（人数/人员类型/诉求） | 所有对话第一步 |
| 2 | `search_policy` | 检索关键词 | RAG 知识库内容 | 需查政策/价格/规则时 |
| 3 | `verify_discount` | 人员类型 + 证件 | 每人优惠等级（免票/半价/全价） | 有特殊人群时 |
| 4 | `calc_ticket_price` | 景区 + 人员详情 + 平台 + 日期 | 费用明细 + 总价 | 需算票价时 |
| 5 | `ticket_query` | 景区 + 日期 + 人数 | 多平台余票 | 用户问"查票"时 |
| 6 | `ticket_book` | 景区 + 日期 + 人数 + 手机号 | 订单确认 | 用户说"订票"时 |
| 7 | `plan_route` | 景区 + 人员类型 + 时长 | 游览路线 | 需路线规划时 |
| 8 | `guide_order_exec` | action + context | 讲解词/预订单/答疑 | 导览/下单时 |

### 2.3 核心调用链

```
parse_user_info  →  search_policy  →  verify_discount  →  calc_ticket_price
      ↓                    ↓                 ↓                   ↓
  识别意图+人员        检索知识库         判定优惠资格          算钱+明细
```

不是每个场景都走完整链。路线规划只需 1→7，导览只需 8。

### 2.4 工具参数容错

`verify_discount` 的容错逻辑（[agent_tools.py:327-350](agent/tools/agent_tools.py)）：
- 检测 LLM 是否错误地将 `person_types` 和 `certificates` 合并成一个 JSON
- 尝试多次解析：原始值 → 剥外层引号 → 剥外层大括号
- 三次都失败才用原始值，不崩溃

### 2.5 模糊匹配的重要性

景区名和人员类型都用了模糊匹配：
- "故宫" → 匹配 "故宫博物院"
- "兵马俑" → 匹配 "秦始皇兵马俑博物馆"
- "老人_70岁" → 匹配 `PERSON_TYPE_DISCOUNTS["老人_70岁以上"]`

因为用户输入和 LLM 传参不会总是精确匹配字典的 key。

## 三、项目落地场景

- 每个 `@tool` 函数被 LangGraph Agent 的 ToolNode 调用
- `calc_ticket_price` 支持 `visit_date` 判断淡旺季，价格自动切换
- `PERSON_TYPE_DISCOUNTS` 保留在代码中（通用逻辑），景区价格从 JSON 加载

## 四、关键代码+逐行注释

### 4.1 工具定义示例：calc_ticket_price
```python
@tool
def calc_ticket_price(
    scenic_spot: str,
    person_details: str = '[{"person_type":"普通成人",...}]',
    platform: str = "meituan",
    visit_date: str = ""  # 可选：用于判断淡旺季
) -> str:
    """根据景区名称、已验证优惠的人员详情、购票平台和游玩日期，
    自动查询基准票价（含淡旺季）、应用折扣和平台附加费..."""

    # 1) 解析人员详情
    details = json.loads(person_details)

    # 2) 模糊匹配景区 → 从 JSON 加载价格数据
    cname, spot_data = _resolve_spot(scenic_spot.strip())

    # 3) 判断淡旺季
    if visit_date:
        month_day = visit_date[5:10]  # "06-15"
        if _in_date_range(month_day, peak["start"], peak["end"]):
            season = "peak"
        else:
            season = "off_peak"
    else:
        season = "peak"  # 无日期默认旺季

    prices = spot_data[f"prices_{season}"]

    # 4) 遍历每人，匹配价格键
    for person in details:
        if "老人" in person["person_type"]:
            base_price = prices["elderly"]
        elif "儿童" in person["person_type"]:
            base_price = prices["child"]
        elif "学生" in person["person_type"]:
            base_price = prices["student"]
        else:
            base_price = prices["adult"]

        # 5) 应用折扣
        discount_amount = round(base_price * person["discount_rate"], 2)
        final_price = round(base_price - discount_amount, 2)

    # 6) 加平台服务费
    total = round(sum(各人实付) + platform_fee, 2)

    return json.dumps({...}, ensure_ascii=False)
```

### 4.2 人员类型映射（PERSON_TYPE_DISCOUNTS）
```python
PERSON_TYPE_DISCOUNTS = {
    "普通成人":   {"discount_type": "全价", "discount_rate": 0.0, "required_docs": ["身份证"]},
    "老人_60至69岁": {"discount_type": "半价", "discount_rate": 0.5, "required_docs": ["身份证"]},
    "老人_70岁以上": {"discount_type": "免票", "discount_rate": 1.0, "required_docs": ["身份证"]},
    "军人_现役":   {"discount_type": "免票", "discount_rate": 1.0, "required_docs": ["军官证或士兵证"]},
    "军人_退伍":   {"discount_type": "免票", "discount_rate": 1.0, "required_docs": ["退伍军人优待证"]},
    "残疾人_1至2级": {"discount_type": "免票", "discount_rate": 1.0, "companion_discount": 0.5},
    "残疾人_3至4级": {"discount_type": "免票", "discount_rate": 1.0},
    "聋哑人士":    {"discount_type": "免票", "discount_rate": 1.0},
    "儿童_6岁以下": {"discount_type": "免票", "discount_rate": 1.0},
    "儿童_6至18岁": {"discount_type": "半价", "discount_rate": 0.5},
    "学生":       {"discount_type": "半价", "discount_rate": 0.5, "required_docs": ["学生证", "身份证"]},
}
```

### 4.3 价格统一加载（_get_pricing）
```python
_pricing_data = None

def _get_pricing():
    """模块级缓存：首次调用从 JSON 加载，后续直接返回缓存"""
    global _pricing_data
    if _pricing_data is None:
        path = os.path.join(os.path.dirname(__file__), '..', '..',
                            'data', 'pricing', 'scenic_spots.json')
        with open(path, 'r', encoding='utf-8') as f:
            _pricing_data = json.load(f)
    return _pricing_data
```

## 五、验收标准+踩坑避坑点

### 5.1 验收标准
- [ ] 8 个工具均能被 `invoke()` 调用并返回有效结果
- [ ] `calc_ticket_price("故宫", ..., visit_date="2025-01-15")` 返回淡季价格
- [ ] `verify_discount` 对"老人"做模糊匹配，能找到半价规则
- [ ] 不存在的景区返回 `{"error": "...", "available_spots": [...]}`

### 5.2 踩坑避坑点
1. **docstring 决定 LLM 行为**：`@tool` 的描述会被注入 system prompt，LLM 据此判断"该不该调这个工具"。描述不清楚 = LLM 调错工具
2. **入参中的 JSON 字符串要容错**：LLM 经常把两个参数打包成一个 JSON，工具代码必须做容错解析（见 `verify_discount` 的三个 fallback）
3. **`_get_pricing()` 是模块级缓存**：文件只读一次。修改 JSON 后需重启应用才能生效
4. **`visit_date` 默认为空 = 旺季**：如果 Agent 没传日期，价格按旺季算。这是有意设计（宁可多算不少算）
