"""
LangGraph 文旅 C 端 Agent
替代原有 LangChain ReAct Agent，提供结构化状态管理和意图路由。
"""

from typing import Annotated, TypedDict, Literal
import json
import re

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from model.factory import chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts
from agent.tools.middleware import reset_runtime_context
from agent.tools.agent_tools import (
    parse_user_info, search_policy, verify_discount, calc_ticket_price,
    plan_route, guide_order_exec, ticket_query, ticket_book,
)


# ============================================================
# State 定义
# ============================================================
class TravelState(TypedDict):
    messages: Annotated[list, add_messages]
    scenic_spot: str
    visit_date: str
    traveler_count: int
    phone: str
    intent: str
    missing_fields: list
    collect_step: int
    final_answer: str


# ============================================================
# 节点函数
# ============================================================

def node_parse_intent(state: TravelState) -> dict:
    """意图解析：调用 parse_user_info 获取意图和人员信息"""
    msgs = state.get("messages", [])
    if not msgs:
        return {"intent": "general"}

    last_msg = msgs[-1]
    query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    try:
        raw = parse_user_info.invoke({"query": query})
        parsed = json.loads(raw)
    except Exception:
        return {"intent": "general"}

    intent = parsed.get("core_need", "general")
    # 意图映射到节点
    intent_map = {
        "ticketing": "ticket_inquiry",       # 查票
        "order_generation": "ticket_booking", # 订票
        "policy_inquiry": "policy_query",     # 优惠政策
        "narration": "narration",             # 导览
        "route_planning": "route_planning",   # 路线
        "credential_verification": "ticket_inquiry",  # 凭证核验→查票
        "general_inquiry": "general",
    }
    mapped = intent_map.get(intent, "general")

    # 提取已知信息
    updates = {
        "intent": mapped,
        "traveler_count": parsed.get("traveler_count", state.get("traveler_count", 1)),
        "collect_step": 0,
    }
    # 从 query 中尝试提取日期
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', query)
    if date_match:
        updates["visit_date"] = date_match.group(1)
    elif "明天" in query:
        import datetime
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        updates["visit_date"] = tomorrow.strftime("%Y-%m-%d")

    logger.info(f"[TravelAgent] intent={mapped}, traveler_count={updates.get('traveler_count')}")
    return updates


def node_collect_info(state: TravelState) -> dict:
    """信息收集：检查必填字段，缺失时生成追问"""
    intent = state.get("intent", "general")
    step = state.get("collect_step", 0)

    # 各意图的必填字段
    required_map = {
        "ticket_inquiry": ["scenic_spot", "visit_date"],
        "ticket_booking": ["scenic_spot", "visit_date", "phone"],
        "route_planning": ["scenic_spot"],
        "policy_query": [],
        "narration": [],
        "general": [],
    }
    required = required_map.get(intent, [])

    # 检查当前状态中缺失的字段
    missing = []
    for field in required:
        val = state.get(field, "")
        if not val or val == "":
            missing.append(field)

    if not missing:
        return {"missing_fields": [], "collect_step": 0}

    # 生成逐个追问（一次只问一个）
    prompt_map = {
        "scenic_spot": "请问您想去哪个景区呢？（如故宫、八达岭长城、黄山等）",
        "visit_date": "请问您计划哪天去呢？（如2026-06-15，或说明天）",
        "phone": "预订需要手机号接收确认短信，请告知您的11位手机号码。",
    }
    current_field = missing[0]
    question = prompt_map.get(current_field, f"请提供{current_field}")

    logger.info(f"[TravelAgent] collect_info: missing={missing}, asking={current_field}")
    return {
        "missing_fields": missing,
        "collect_step": step + 1,
        "final_answer": question,
    }


def node_ticket_inquiry(state: TravelState) -> dict:
    """查票节点：调用 ticket_query"""
    scenic = state.get("scenic_spot", "")
    date = state.get("visit_date", "")
    count = state.get("traveler_count", 1)

    if not scenic or not date:
        missing = []
        if not scenic:
            missing.append("scenic_spot")
        if not date:
            missing.append("visit_date")
        return {"missing_fields": missing, "collect_step": 1}

    try:
        result = ticket_query.invoke({
            "scenic_spot": scenic,
            "visit_date": date,
            "traveler_count": count,
        })
    except Exception as e:
        result = f"查询余票时遇到问题：{e}，请稍后再试。"

    return {"final_answer": result}


def node_ticket_booking(state: TravelState) -> dict:
    """订票节点：调用 ticket_book"""
    scenic = state.get("scenic_spot", "")
    date = state.get("visit_date", "")
    count = state.get("traveler_count", 1)
    phone = state.get("phone", "")

    try:
        result = ticket_book.invoke({
            "scenic_spot": scenic,
            "visit_date": date,
            "traveler_count": count,
            "phone": phone,
        })
    except Exception as e:
        result = f"预订时遇到问题：{e}，请稍后再试。"

    return {"final_answer": result}


def node_policy_query(state: TravelState) -> dict:
    """政策查询节点：search_policy → verify_discount → calc_ticket_price"""
    msgs = state.get("messages", [])
    query = msgs[-1].content if msgs else ""

    try:
        raw = parse_user_info.invoke({"query": query})
        parsed = json.loads(raw)
    except Exception:
        return {"final_answer": "抱歉，无法解析您的查询，请重新描述一下您的问题。"}

    # 如果信息不足，生成追问
    if parsed.get("needs_info"):
        return {"final_answer": parsed.get("missing_info_prompt", "请补充年龄和景区信息。")}

    # 检索政策
    policy_query_str = f"{state.get('scenic_spot', '景区')} 优惠政策 老人 儿童 军人 残疾人"
    policy_result = search_policy.invoke({"query": policy_query_str})

    # 人员类型 + 优惠判定
    person_types = parsed.get("person_types", ["普通成人"])
    pt_str = json.dumps(person_types, ensure_ascii=False)
    verify_result = verify_discount.invoke({
        "person_types": pt_str,
        "certificates": "{}",
    })

    # 票价计算
    scenic = state.get("scenic_spot", "故宫")
    price_result = calc_ticket_price.invoke({
        "scenic_spot": scenic,
        "person_details": verify_result,
        "platform": "meituan",
    })

    # 组装最终回答
    try:
        price_data = json.loads(price_result)
        total = price_data.get("total_price", "?")
        breakdown = price_data.get("price_breakdown", [])
    except Exception:
        total = "?"
        breakdown = []

    summary_lines = [f"根据查询结果，{scenic}的票价信息如下："]
    for item in breakdown:
        ptype = item.get("person_type", "")
        final = item.get("final_price", "?")
        reason = item.get("discount_reason", "")
        summary_lines.append(f"  · {ptype}：{final}元 ({reason})" if reason else f"  · {ptype}：{final}元")
    summary_lines.append(f"  总价约：{total}元（以景区当日实际价格为准）")

    return {"final_answer": "\n".join(summary_lines)}


def node_route_planning(state: TravelState) -> dict:
    """路线规划节点"""
    scenic = state.get("scenic_spot", "故宫")
    count = state.get("traveler_count", 1)
    msgs = state.get("messages", [])
    query = msgs[-1].content if msgs else ""

    # 从 parse_user_info 获取人员类型
    try:
        raw = parse_user_info.invoke({"query": query})
        parsed = json.loads(raw)
        ttypes = parsed.get("person_types", ["普通成人"])
    except Exception:
        ttypes = ["普通成人"]

    try:
        result = plan_route.invoke({
            "scenic_spot": scenic,
            "traveler_types": json.dumps(ttypes, ensure_ascii=False),
            "duration_hours": 4.0,
        })
    except Exception as e:
        result = f"路线规划遇到问题：{e}"

    return {"final_answer": result}


def node_narration(state: TravelState) -> dict:
    """导览节点"""
    msgs = state.get("messages", [])
    query = msgs[-1].content if msgs else ""

    # 从 query 中提取地点名
    from agent.tools.agent_tools import POINT_TO_SPOT
    spot = ""
    for point_name in POINT_TO_SPOT:
        if point_name in query:
            spot = point_name
            break
    if not spot:
        spot = query.replace("介绍一下", "").replace("介绍", "").replace("讲解", "").strip()

    try:
        result = guide_order_exec.invoke({
            "action": "narration",
            "context": json.dumps({"scenic_spot": spot}),
        })
    except Exception as e:
        result = f"导览服务遇到问题：{e}"

    return {"final_answer": result}


def node_general(state: TravelState) -> dict:
    """通用回复节点"""
    msgs = state.get("messages", [])
    query = msgs[-1].content if msgs else ""

    system_prompt = load_system_prompts()
    llm_msgs = [
        {"role": "system", "content": system_prompt[:2000]},
    ]
    for m in msgs[-4:]:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        content = m.content if hasattr(m, "content") else str(m)
        llm_msgs.append({"role": role, "content": content})

    try:
        response = chat_model.invoke(llm_msgs)
        return {"final_answer": response.content}
    except Exception as e:
        return {"final_answer": f"抱歉，暂时无法处理您的请求：{e}"}


# ============================================================
# 路由函数
# ============================================================

def route_after_parse(state: TravelState) -> str:
    """解析意图后路由"""
    intent = state.get("intent", "general")
    missing = state.get("missing_fields", [])

    if missing:
        return "collect_info"
    return intent


def route_after_collect(state: TravelState) -> str:
    """信息收集后路由"""
    missing = state.get("missing_fields", [])
    if missing:
        return END  # 返回追问给用户
    return state.get("intent", "general")


# ============================================================
# 构建 Graph
# ============================================================

def build_travel_graph() -> StateGraph:
    """构建 LangGraph 文旅 Agent"""
    workflow = StateGraph(TravelState)

    # 添加节点
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

    # parse_intent 后的分支
    workflow.add_conditional_edges(
        "parse_intent",
        route_after_parse,
        {
            "collect_info": "collect_info",
            "ticket_inquiry": "ticket_inquiry",
            "ticket_booking": "ticket_booking",
            "policy_query": "policy_query",
            "route_planning": "route_planning",
            "narration": "narration",
            "general": "general",
        }
    )

    # collect_info 后
    workflow.add_conditional_edges(
        "collect_info",
        route_after_collect,
        {
            "general": "general",  # fallback
        }
    )
    workflow.add_edge("collect_info", END)  # missing fields → end (return question)

    # 各业务节点 → END
    workflow.add_edge("ticket_inquiry", END)
    workflow.add_edge("ticket_booking", END)
    workflow.add_edge("policy_query", END)
    workflow.add_edge("route_planning", END)
    workflow.add_edge("narration", END)
    workflow.add_edge("general", END)

    return workflow.compile()


# ============================================================
# Agent 封装（兼容 app.py 流式接口）
# ============================================================

class TravelAgent:
    """LangGraph 文旅 Agent，兼容 Streamlit 流式输出"""

    def __init__(self):
        self.graph = build_travel_graph()
        self._state = {
            "messages": [],
            "scenic_spot": "",
            "visit_date": "",
            "traveler_count": 1,
            "phone": "",
            "intent": "",
            "missing_fields": [],
            "collect_step": 0,
            "final_answer": "",
        }

    def _extract_from_text(self, text: str):
        """从用户输入中提取结构化信息并更新状态"""
        # 提取景区名
        spots = ["故宫", "八达岭长城", "慕田峪长城", "长城", "黄山", "西湖", "兵马俑", "颐和园", "张家界", "九寨沟", "布达拉宫", "漓江", "桂林"]
        for s in spots:
            if s in text:
                self._state["scenic_spot"] = s
                break
        # 提取日期
        import datetime
        if "明天" in text:
            tomorrow = datetime.date.today() + datetime.timedelta(days=1)
            self._state["visit_date"] = tomorrow.strftime("%Y-%m-%d")
        elif "后天" in text:
            after = datetime.date.today() + datetime.timedelta(days=2)
            self._state["visit_date"] = after.strftime("%Y-%m-%d")
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
        if date_match:
            self._state["visit_date"] = date_match.group(1)
        # 提取人数
        count_match = re.search(r'(\d+)\s*[个位名]', text)
        if count_match:
            self._state["traveler_count"] = int(count_match.group(1))
        # 提取手机号
        phone_match = re.search(r'1[3-9]\d{9}', text)
        if phone_match:
            self._state["phone"] = phone_match.group(0)

    def execute_stream(self, query: str, chat_history: str = ""):
        """流式执行，兼容原 ReactAgent 接口"""
        reset_runtime_context()

        # 更新状态——从用户输入提取信息
        self._extract_from_text(query)

        # 添加消息
        self._state["messages"].append(HumanMessage(content=query))

        # 从 chat_history 重建上下文
        if chat_history:
            for line in chat_history.split("\n"):
                if line.startswith("用户: "):
                    hist_query = line[4:]
                    self._extract_from_text(hist_query)

        # 执行 graph
        try:
            result = self.graph.invoke(self._state)
            answer = result.get("final_answer", "")

            # 处理 collect_info 返回的追问
            if not answer and result.get("missing_fields"):
                field = result["missing_fields"][0]
                prompts = {
                    "scenic_spot": "请问您想去哪个景区呢？（如故宫、八达岭长城等）",
                    "visit_date": "请问您计划哪天去呢？（如2026-06-15，或说明天）",
                    "phone": "预订需要手机号接收确认短信，请告知您的11位手机号码。",
                }
                answer = prompts.get(field, f"请提供{field}")

            yield answer + "\n"
        except Exception as e:
            logger.error(f"[TravelAgent] 执行异常：{e}")
            yield f"抱歉，处理您的请求时遇到了问题：{e}\n"


if __name__ == '__main__':
    agent = TravelAgent()
    for chunk in agent.execute_stream("帮我查一下故宫明天的余票，3个人"):
        print(chunk, end="", flush=True)
