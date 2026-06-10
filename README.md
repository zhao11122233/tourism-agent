# 🏯 文旅智能助手 — 多平台票务聚合与导览全链路智能体

## 项目概述

基于 LangChain ReAct Agent 架构，结合 ChromaDB 向量知识库与 RAG 检索增强生成技术，构建面向文旅场景的 AI 全链路智能体。覆盖**需求咨询 → 优惠判定 → 价格计算 → 下单出单 → 路线规划 → 沿路交互式导览 → 多凭证统一入园核验**完整业务闭环。

打通美团、携程、景区自营多平台数据，对用户身份证、手机号、短信验证码、各平台订单凭证做统一聚合归一，实现一次数据聚合、多方式入园核验。

---

## 核心特性

#### 1. 7 工具 ReAct Agent
- Agent 通过「思考→行动→观察→再思考」循环自主编排工具调用链
- 7 个工具：信息解析、政策检索、优惠判定、票价计算、路线规划、导览执行、多平台凭证聚合核验

#### 2. 多平台统一凭证聚合
- 美团 / 携程 / 景区自营三平台订单数据统一关联
- 支持身份证、手机号、短信验证码、二维码任一凭证入园核验
- 订单状态跨平台同步

#### 3. 特殊人群优惠智能判定
- 覆盖老人（60-69半价/70+免票）、儿童（6岁以下免票/6-18半价）、军人（现役/退伍/消防 免票）、残疾人（1-4级 免票+陪护优惠）、聋哑人士（免票）
- 自动核验证件类型与优惠资格

#### 4. RAG 知识库检索
- 6 大文旅知识领域：多平台订单结构、票务价格规则、特殊群体优惠政策、入园凭证核验规则、景区导览讲解材料、短信验证码格式
- ChromaDB 向量存储，Top-3 召回，LLM 总结生成回答

#### 5. 交互式景点导览
- 支持 80+ 景区细分点位（故宫午门/长城北一楼/黄山迎客松等）
- 自动点位→景区关联，内置讲解词 + RAG 知识库回退
- 导览意图强约束，绝不混杂票务信息

#### 6. Streamlit 流式对话界面
- 打字机效果逐字输出，模拟真人对话体验
- 上下文记忆 + 场景化追问白名单 + 短词意图继承

---

## 项目结构

```bash
.
├── agent/                           # Agent 核心逻辑
│   ├── react_agent.py               # ReAct Agent 主逻辑
│   └── tools/
│       ├── agent_tools.py           # 7 个文旅工具 + 内置模拟数据
│       └── middleware.py            # 工具监控 & 运行时上下文
├── config/                          # YAML 配置文件
│   ├── agent.yml                    # 模拟数据路径
│   ├── chroma.yml                   # 向量库参数
│   ├── prompts.yml                  # 提示词路径映射
│   └── rag.yml                      # 模型名称
├── data/
│   ├── knowledge/                   # 知识库文档（6 个 .txt）
│   │   ├── 多平台订单数据结构.txt
│   │   ├── 票务价格规则.txt
│   │   ├── 特殊群体优惠政策.txt
│   │   ├── 入园凭证核验规则.txt
│   │   ├── 景区导览讲解材料.txt
│   │   └── 短信验证码格式.txt
│   └── tourism/                     # 模拟数据（5 个 JSON/CSV）
│       ├── orders.json              # 跨平台订单样例
│       ├── scenic_spots.csv         # 10+ 景区基础信息
│       ├── discounts.json           # 优惠规则结构
│       ├── routes.json              # 路线模板
│       └── narrations.json          # 讲解素材
├── model/                           # 模型工厂
│   └── factory.py                   # Qwen3-Max & Embedding 初始化
├── prompts/                         # 提示词模板
│   ├── main_prompt.txt              # ReAct 系统提示词
│   ├── rag_summarize.txt            # RAG 总结提示词
│   └── order_output_prompt.txt      # 订单生成提示词
├── rag/                             # RAG 检索增强模块
│   ├── rag_service.py               # 检索 + 总结服务
│   └── vector_store.py              # Chroma 向量库 & 文档加载
├── utils/                           # 通用工具
│   ├── config_handler.py            # YAML 配置加载
│   ├── file_handler.py              # 文件处理 & MD5
│   ├── logger_handler.py            # 日志管理
│   ├── path_tool.py                 # 路径工具
│   └── prompt_loader.py             # 提示词加载
├── app.py                           # Streamlit 应用入口
├── requirements.txt
└── README.md
```

---

## 7 个工具能力总览

| # | 工具 | 功能 |
|---|------|------|
| 1 | `parse_user_info` | 解析出行人数、人员类型（老人/儿童/军人/残疾人/聋哑）、核心诉求 |
| 2 | `search_policy` | RAG 检索优惠规则、票务政策、入园流程、导览材料 |
| 3 | `verify_discount` | 判定每类人员优惠等级（免票/半价/全价）+ 所需证件 |
| 4 | `calc_ticket_price` | 查基准票价 → 应用折扣 → 计算最优总价 |
| 5 | `plan_route` | 生成分段式游览路线（节点/时长/无障碍标注） |
| 6 | `guide_order_exec` | 景点讲解/互动答疑/预订单生成 |
| 7 | `aggregate_verify_credentials` | 跨平台订单聚合 → 统一身份 → 入园通行判定 |

---

## 业务全链路

```
需求咨询 → 优惠判定 → 价格计算 → 下单出单 → 路线规划 → 沿路交互式导览 → 多凭证统一入园核验
```

---

## 快速开始

### 环境要求
- Python 3.10 及以上
- 阿里云百炼 DashScope API Key

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/zhao11122233/tourism-agent.git
cd tourism-agent

# 安装依赖
pip install -r requirements.txt

# 配置 API Key（在项目根目录创建 .env 文件）
echo DASHSCOPE_API_KEY=your-api-key > .env

# 构建向量知识库（首次运行）
python -m rag.vector_store

# 启动应用
python -m streamlit run app.py
```

---

## 演示问题示例

#### 票务优惠咨询
- 我有2个老人和1个儿童，想去故宫游玩，帮我看看有什么优惠政策
- 两个大人一个小孩，小孩5岁，能免门票吗？
- 帮我算一下带军人证去黄山要花多少钱

#### 路线规划
- 帮我规划故宫一日游路线，有老人在
- 兵马俑半天怎么玩

#### 景点导览
- 我现在到了北一楼，介绍一下这个景点
- 午门有什么历史故事？

#### 凭证核验
- 用手机号138****5678核验我的故宫订单
- 我在美团和携程都买了票，怎么统一入园

#### 综合全链路
- 我想带家人（2成人1老人1儿童）去八达岭长城，帮我从购票到入园全部搞定

---

## 配置说明

| 文件 | 说明 |
|------|------|
| `config/agent.yml` | 模拟数据路径（订单/景区/优惠/路线/讲解） |
| `config/chroma.yml` | 向量库、分块策略、检索参数 |
| `config/prompts.yml` | 三套提示词模板路径映射 |
| `config/rag.yml` | 模型名称（qwen3-max / text-embedding-v4） |

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 大模型 | 阿里通义千问 Qwen3-Max |
| Agent 框架 | LangChain ReAct Agent + AgentExecutor |
| 向量检索 | ChromaDB + DashScope text-embedding-v4 |
| 前端界面 | Streamlit（流式对话） |
| 配置管理 | YAML + python-dotenv |

---

## 感谢与支持
- LangChain / LangGraph
- Streamlit
- ChromaDB
- 阿里云百炼 DashScope

---

### ⭐ Final
本项目仅供学习与交流，如果觉得有帮助，欢迎点个 Star！
