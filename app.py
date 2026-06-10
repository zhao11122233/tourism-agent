import time
import os

# 禁用Streamlit的使用统计和邮箱提示
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

import streamlit as st
from agent.react_agent import ReactAgent

# 标题
st.title("文旅智能助手 — 多平台票务聚合与导览服务")
st.divider()

# 侧边栏 — 示例查询
with st.sidebar:
    st.subheader("示例查询")
    st.markdown("""
**票务优惠类：**
- 我有2个老人和1个儿童，想去故宫游玩，帮我看看有什么优惠政策
- 帮我算一下带军人证去黄山要花多少钱

**路线规划类：**
- 帮我规划故宫一日游路线，有老人在
- 兵马俑半天怎么玩

**凭证核验类：**
- 用手机号138****5678核验我的故宫订单
- 我在美团和携程都买了票，怎么统一入园

**综合全链路：**
- 我想带家人（2成人1老人1儿童）去八达岭长城，帮我从购票到入园全部搞定
""")

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    avatar = "🧳" if message["role"] == "user" else "🏯"
    st.chat_message(message["role"], avatar=avatar).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user", avatar="🧳").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 构建历史对话上下文（只取最近3轮，避免跨对话污染）
    history_parts = []
    recent_spot = ""
    recent_intent = ""
    intent_words = ["票", "多少钱", "优惠", "免票", "买票", "购票", "价格", "订票", "能不能", "需要买", "要买票"]
    # 检测"人数+年龄"模式——隐含票务意图
    def has_ticket_intent(text):
        if any(w in text for w in intent_words):
            return True
        has_people = any(w in text for w in ["人", "大人", "小孩", "儿童", "老人", "成人"])
        has_age = "岁" in text
        return has_people and has_age  # "两大一小，孩子4岁" → 票务意图
    for msg in st.session_state["message"][-6:]:
        role_label = "用户" if msg["role"] == "user" else "助手"
        history_parts.append(f"{role_label}: {msg['content']}")
        if msg["role"] == "user":
            for spot in ["故宫", "长城", "黄山", "西湖", "兵马俑", "颐和园", "张家界", "九寨沟", "布达拉宫", "漓江", "八达岭"]:
                if spot in msg["content"]:
                    recent_spot = spot
            if has_ticket_intent(msg["content"]):
                recent_intent = "票务咨询"
    chat_history = "\n".join(history_parts)

    # 短追问：补充上下文（仅当当前query很短且无完整意图时）
    enriched_prompt = prompt
    if len(prompt) <= 10 and not has_ticket_intent(prompt):
        hints = []
        if recent_spot and not any(s in prompt for s in ["故宫", "长城", "黄山", "西湖", "兵马俑", "颐和园", "张家界", "九寨沟", "布达拉宫", "漓江", "八达岭"]):
            hints.append(f"景区是{recent_spot}")
        if recent_intent:
            hints.append(f"用户之前的意图是{recent_intent}")
        if hints:
            enriched_prompt = f"{prompt}（{'，'.join(hints)}，请基于此回答，不要转入导览模式）"

    response_messages = []
    status = st.status("文旅助手思考中...", expanded=False)

    def capture(generator, cache_list):
        for chunk in generator:
            cache_list.append(chunk)
            for char in chunk:
                time.sleep(0.01)
                yield char

    res_stream = st.session_state["agent"].execute_stream(enriched_prompt, chat_history)
    st.chat_message("assistant", avatar="🏯").write_stream(capture(res_stream, response_messages))
    status.update(label="已完成", state="complete")
    st.session_state["message"].append({"role": "assistant", "content": response_messages[-1] if response_messages else ""})
    if response_messages:
        st.rerun()
