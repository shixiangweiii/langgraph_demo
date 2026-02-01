'''
傻逼通义给的human-in-loop demo案例代码
'''
import os
from operator import add
from typing import TypedDict, Annotated

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph

llm = ChatTongyi(model="qwen-max", api_key=os.getenv("DASHSCOPE_API_KEY"))

# ================= STATE =================
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    task_type: str
    awaiting_human_input: bool


# ================= HUMAN NODE（占位）=================
def human_feedback_node(state: State):
    return {}


# ================= SUPERVISOR =================
def supervisor_node(state: State):
    writer = get_stream_writer()

    # 🟥 如果正在等人工 → 真正暂停图执行
    if state.get("awaiting_human_input"):
        writer({"supervisor": "⏸ 等待人工输入..."})
        return {"__interrupt__": "waiting for human input"}

    last_msg = state["messages"][-1].content

    # 🟥 高风险检测
    sensitive_keywords = ["弄死", "杀死", "爆炸", "违法", "攻击"]
    if any(kw in last_msg for kw in sensitive_keywords):
        writer({"alert": "检测到高风险内容，进入人工审核"})
        return {
            "awaiting_human_input": True,
            "__interrupt__": "需要人工审核"
        }

    # 🟩 没任务类型才分类
    if not state.get("task_type"):
        writer({"supervisor": "进行任务分类"})
        prompt = """分类用户请求：
        旅游路线规划 -> travel
        讲一个笑话 -> joke
        对一个对联 -> couplet
        其他 -> other
        只返回英文单词"""
        response = llm.invoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": last_msg}
        ])
        task_type = response.content.strip().lower()
        writer({"supervisor": f"分类结果: {task_type}"})
        return {"task_type": task_type}

    # 🟩 子任务执行完成 → 结束
    writer({"supervisor": "任务完成"})
    return {"task_type": END}


# ================= 子任务 =================
def travel_node(state: State):
    return {"messages": [AIMessage(content="推荐杭州三日游：西湖、灵隐寺、千岛湖。")]}

def joke_node(state: State):
    return {"messages": [AIMessage(content="为什么程序员总分不清万圣节和圣诞节？因为 Oct 31 == Dec 25 😄")]}

def couplet_node(state: State):
    return {"messages": [AIMessage(content="上联：山高水远情长在；下联：月白风清意自闲。")]}

def other_node(state: State):
    return {"messages": [AIMessage(content="抱歉，这个问题我暂时无法处理。")]}


# ================= ROUTER =================
def router(state: State):
    if state.get("awaiting_human_input"):
        return "human_feedback_node"

    if state.get("task_type") == END:
        return END

    if state.get("task_type") in {"travel", "joke", "couplet", "other"}:
        return f"{state['task_type']}_node"

    return "supervisor_node"


# ================= GRAPH =================
builder = StateGraph(State)

builder.add_node("supervisor_node", supervisor_node)
builder.add_node("human_feedback_node", human_feedback_node)
builder.add_node("travel_node", travel_node)
builder.add_node("joke_node", joke_node)
builder.add_node("couplet_node", couplet_node)
builder.add_node("other_node", other_node)

builder.add_edge(START, "supervisor_node")

builder.add_conditional_edges(
    "supervisor_node",
    router,
    ["travel_node", "joke_node", "couplet_node", "other_node", "human_feedback_node", END]
)

builder.add_edge("travel_node", "supervisor_node")
builder.add_edge("joke_node", "supervisor_node")
builder.add_edge("couplet_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")

# human 节点回 supervisor（但不会死循环，因为 supervisor 会 interrupt）
builder.add_edge("human_feedback_node", "supervisor_node")

graph = builder.compile(checkpointer=InMemorySaver())


# ================= 主程序 =================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo"}}
    user_input = "我要弄死你"

    print("🚀 用户输入:", user_input)

    for chunk in graph.stream(
        {"messages": [HumanMessage(content=user_input)], "task_type": "", "awaiting_human_input": False},
        config=config,
        stream_mode="custom"
    ):
        print("▶️", chunk)

    state = graph.get_state(config)

    # 如果被中断
    if state.values.get("awaiting_human_input"):
        print("\n🧑 人工介入中...")
        correction = "我想去杭州旅游，请规划路线"

        graph.update_state(
            config,
            {
                "messages": [HumanMessage(content=correction)],
                "awaiting_human_input": False,
                "task_type": ""
            }
        )

        print("\n🔁 恢复执行...\n")
        for chunk in graph.stream(None, config=config, stream_mode="custom"):
            print("▶️", chunk)

    print("\n最终消息记录:")
    for m in graph.get_state(config).values["messages"]:
        role = "👤" if isinstance(m, HumanMessage) else "🤖"
        print(role, m.content)
