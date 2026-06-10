# 文旅智能助手 — 查票 · 订票 · 优惠 · 路线 · 导览

## 项目概述

基于 **LangGraph** 状态管理 + ChromaDB 向量知识库 + RAG 检索增强生成技术，构建面向 **C 端游客** 的文旅全链路智能体。覆盖 **需求咨询 → 查票 → 订票 → 优惠判定 → 价格计算 → 路线规划 → 交互式导览** 完整业务闭环。

独立客户端层封装第三方票务 API，Agent 层专注对话交互与意图路由，两层严格分离。

---

## 核心特性

#### 1. LangGraph 意图路由
- 会话状态管理（景区、日期、人数、手机号），自动提取上下文信息
- 意图识别 + 条件路由：查票 / 订票 / 优惠政策 / 路线规划 / 景点导览
- 信息缺项时逐个追问，已获取信息不重复索要

#### 2. 查票与订票
- `ticket_query`：查询美团 / 携程 / 景区官方多平台余票与票价
- `ticket_book`：提交门票预订，手机号校验，返回订单确认
- 独立 `client/ticket_client.py` 封装网络请求，Agent 层零网络依赖

#### 3. 特殊人群优惠智能判定
- 老人（60-69 半价 / 70+ 免票）、儿童（6 岁以下免票 / 6-18 半价）、军人（现役/退伍 免票）、残疾人（1-4 级免票 + 陪护优惠）、聋哑人士（免票）
- 自动匹配优惠规则，输出所需证件

#### 4. RAG 知识库检索
- 6 大文旅知识领域：多平台订单结构、票务价格规则、特殊群体优惠政策、入园凭证核验规则、景区导览讲解材料、短信验证码格式
- ChromaDB 向量存储，Top-3 召回，LLM 总结生成回答

#### 5. 交互式景点导览
- 80+ 景区细分点位（故宫午门 / 长城北一楼 / 黄山迎客松等）
- 自动点位→景区关联，内置讲解词 + RAG 知识库回退

#### 6. Streamlit 流式对话
- 上下文记忆 + 场景化追问 + 参数格式校验 + 手机号隐私保护

---

## 项目结构

```bash
.
├── agent/                           # Agent 核心逻辑
│   ├── travel_agent.py              # ★ LangGraph 状态图 Agent（主）
│   ├── react_agent.py               # [保留] 原 LangChain ReAct Agent
│   └── tools/
│       ├── agent_tools.py           # 8 个文旅工具 + 内置模拟数据
│       └── middleware.py            # 工具监控 & 运行时上下文
├── client/                          # ★ 独立票务客户端层
│   └── ticket_client.py             # 余票查询 / 门票预订 (HTTP 封装)
├── config/                          # YAML 配置文件
│   ├── agent.yml                    # 数据路径 & 客户端配置
│   ├── chroma.yml                   # 向量库参数
│   ├── prompts.yml                  # 提示词路径映射
│   └── rag.yml                      # 模型名称
├── data/
│   ├── knowledge/                   # 知识库文档（6 个 .txt）
│   └── tourism/                     # 模拟数据（5 个 JSON/CSV）
├── model/factory.py                 # Qwen3-Max & Embedding 初始化
├── prompts/                         # 提示词模板（3 个 .txt）
├── rag/                             # RAG 检索增强模块
├── utils/                           # 通用工具
├── app.py                           # Streamlit 应用入口
├── requirements.txt
└── README.md
```

---

## 8 个工具总览

| # | 工具 | 功能 | 场景 |
|---|------|------|------|
| 1 | `parse_user_info` | 解析人数、人员类型、核心诉求 | 所有对话入口 |
| 2 | `search_policy` | RAG 检索优惠规则、票务政策、导览材料 | 政策查询 |
| 3 | `verify_discount` | 判定优惠等级（免票/半价/全价）+ 所需证件 | 优惠判定 |
| 4 | `calc_ticket_price` | 查基准票价 → 应用折扣 → 计算总价 | 价格计算 |
| 5 | `ticket_query` | 查询多平台余票和票价 | 查票 |
| 6 | `ticket_book` | 提交门票预订，手机号校验，返回订单确认 | 订票 |
| 7 | `plan_route` | 生成分段式游览路线（节点/时长/无障碍） | 路线规划 |
| 8 | `guide_order_exec` | 景点讲解 / 互动答疑 / 预订单生成 | 导览 |

---

## 分层架构

```
┌─────────────────────────────────┐
│  Streamlit UI (app.py)          │  ← 对话界面
├─────────────────────────────────┤
│  LangGraph Agent                │  ← 意图路由、状态管理、参数校验
│  (agent/travel_agent.py)        │
├─────────────────────────────────┤
│  8 Tools (agent_tools.py)       │  ← 工具调用、结果转自然语言
├─────────────────────────────────┤
│  TicketClient (client/)         │  ← HTTP 请求、超时、重试、异常兜底
├─────────────────────────────────┤
│  ChromaDB RAG + Qwen3-Max LLM   │  ← 知识检索 + 生成
└─────────────────────────────────┘
```

Agent 层 **不处理** 网络请求、接口鉴权、签名、密钥 — 全部在 TicketClient 层隔离。

---

## 业务全链路

```
需求咨询 → 查票/订票 → 优惠判定 → 价格计算 → 路线规划 → 交互式导览
```

---

## 快速开始

### 环境要求
- Python 3.10 及以上
- 阿里云百炼 DashScope API Key

```bash
git clone https://github.com/zhao11122233/tourism-agent.git
cd tourism-agent
pip install -r requirements.txt

# 配置 API Key
echo DASHSCOPE_API_KEY=your-api-key > .env

# 构建向量知识库
python -m rag.vector_store

# 启动
python -m streamlit run app.py
```

---

## 演示示例

#### 查票 / 订票
- 帮我查一下故宫明天的余票，3个人
- 帮我订八达岭长城2026-06-15的票，2个人，手机号13812345678

#### 票务优惠
- 我有2个老人和1个儿童，想去故宫，有什么优惠政策
- 两个大人一个小孩，小孩5岁能免门票吗
- 帮我算一下带军人证去黄山要花多少钱

#### 路线规划
- 帮我规划故宫一日游路线，有老人在
- 兵马俑半天怎么玩

#### 景点导览
- 我现在到了北一楼，介绍一下这个景点
- 午门有什么历史故事

---

## 配置说明

| 文件 | 说明 |
|------|------|
| `config/agent.yml` | 模拟数据路径 + 票务客户端配置 |
| `config/chroma.yml` | 向量库、分块策略、检索参数 |
| `config/prompts.yml` | 提示词路径映射 |
| `config/rag.yml` | 模型名称（qwen3-max / text-embedding-v4） |

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 大模型 | 阿里通义千问 Qwen3-Max |
| Agent 框架 | LangGraph StateGraph |
| 向量检索 | ChromaDB + DashScope text-embedding-v4 |
| 前端界面 | Streamlit（流式对话） |
| 配置管理 | YAML + python-dotenv |

---

## 后续优化方向

> 以下为规划中的能力扩展，暂未实现。

#### 👐 聋哑人士手语服务
- 集成手语识别与合成能力，为聋哑游客提供无障碍导览体验

#### 🤖 多 Agent 全行程规划
- **酒店 Agent**：根据目的地、预算、偏好自动推荐住宿方案
- **航班 Agent**：整合机票、高铁、景区摆渡车等多模式交通规划
- **活动 Agent**：基于目的地推荐当地演出、节庆、美食等体验活动

---

## 感谢与支持
- LangChain / LangGraph
- Streamlit
- ChromaDB
- 阿里云百炼 DashScope

---

### ⭐ Final
本项目仅供学习与交流，如果觉得有帮助，欢迎点个 Star！
