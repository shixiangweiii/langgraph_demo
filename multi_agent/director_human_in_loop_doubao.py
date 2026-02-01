import os
from operator import add
from typing import TypedDict, Annotated

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph
import logging

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app_with_human_loop.log')
    ]
)

# 初始化通义千问LLM（请确保环境变量LLM_SK已配置）
llm = ChatTongyi(model="qwen3-max-preview", api_key=os.getenv("LLM_SK"))


# 扩展全局状态：增加人工审核相关字段
class State(TypedDict):
    # 消息列表（追加策略）
    messages: Annotated[list[AnyMessage], add]
    # 任务类型（travel/joke/couplet/other）
    type: str
    # 人工审核状态（pending/approved/rejected/modified）
    human_approval: str
    # 是否等待人工输入（标记是否需要暂停流程）
    pending_human_input: bool


# -------------------------- 原有节点（微调） --------------------------
def supervisor_node(state: State):
    """监督节点：任务分类 + 流程结束判断"""
    logging.info(f"supervisor_node - type={state.get('type', '')}, human_approval={state.get('human_approval', '')}")
    writer = get_stream_writer()

    # 1. 首次执行：无type，进行任务分类
    if not state.get("type"):
        writer({"node": "开始任务规划（首次执行）"})
        prompt = """你是专业客服助手，负责分类用户问题：
                    旅游路线规划 → travel；讲笑话 → joke；对对联 → couplet；其他 → other；
                    仅返回上述关键词，不要其他内容。
                 """
        prompts = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": state["messages"][-1].content},
        ]
        response = llm.invoke(prompts)
        type_res = response.content.strip()
        writer({"supervisor_step": f"任务分类结果：{type_res}"})

        # 校验分类结果
        if type_res not in ["travel", "joke", "couplet", "other"]:
            raise ValueError(f"NLU分类失败：{type_res}")
        return {"type": type_res, "pending_human_input": False}

    # 2. 人工审核通过：结束流程
    elif state.get("human_approval") == "approved":
        writer({"supervisor_step": f"人工已批准，任务 {state['type']} 完成，流程结束"})
        return {"type": END, "pending_human_input": False}

    # 3. 人工审核拒绝：重新执行Worker节点
    elif state.get("human_approval") == "rejected":
        writer({"supervisor_step": f"人工拒绝，重新执行 {state['type']} 任务"})
        return {"human_approval": "", "pending_human_input": False}

    # 4. 人工修改：更新消息后重新执行Worker节点
    elif state.get("human_approval") == "modified":
        writer({"supervisor_step": "人工已修改需求，重新执行任务"})
        return {"human_approval": "", "pending_human_input": False}

    # 5. Worker执行完成：标记需要人工审核
    else:
        writer({"supervisor_step": "Worker节点执行完成，等待人工审核"})
        return {"pending_human_input": True}


def travel_node(state: State):
    """旅游规划Worker节点"""
    writer = get_stream_writer()
    writer({"node": "执行旅游规划任务"})
    # 模拟生成旅游规划结果（实际场景可调用LLM生成真实内容）
    response_content = "为你规划的旅游路线：北京→上海→杭州，全程5天，主打人文+自然风光。"
    return {
        "messages": [AIMessage(content=response_content)],
        "pending_human_input": False
    }


def joke_node(state: State):
    """讲笑话Worker节点"""
    writer = get_stream_writer()
    writer({"node": "执行讲笑话任务"})
    response_content = "为什么程序员喜欢用深色模式？因为他们不想让代码看到自己的黑眼圈😜。"
    return {
        "messages": [AIMessage(content=response_content)],
        "pending_human_input": False
    }


def couplet_node(state: State):
    """对对联Worker节点"""
    writer = get_stream_writer()
    writer({"node": "执行对对联任务"})
    response_content = "上联：春风送暖入屠苏；下联：瑞雪迎春到人间；横批：万象更新。"
    return {
        "messages": [AIMessage(content=response_content)],
        "pending_human_input": False
    }


def other_node(state: State):
    """兜底节点"""
    writer = get_stream_writer()
    writer({"node": "执行兜底处理"})
    response_content = "未知问题，我暂时无法处理这个任务。"
    return {
        "messages": [AIMessage(content=response_content)],
        "pending_human_input": False
    }


# -------------------------- 新增：人工审核逻辑（独立函数） --------------------------
def handle_human_approval(current_state):
    """处理人工审核（独立函数，供外部调用）"""
    # 打印当前Worker执行结果，供人类参考
    latest_ai_msg = next((msg for msg in reversed(current_state["messages"]) if isinstance(msg, AIMessage)), None)
    if latest_ai_msg:
        print(f"\n===== 待审核内容 =====\n{latest_ai_msg.content}\n")

    # 提示人类输入指令
    print("请输入审核指令（仅支持：approved/批准 | rejected/拒绝 | modified/修改+新需求）：")
    human_input = input("> ").strip().lower()

    # 解析人工指令
    if human_input in ["approved", "批准"]:
        return {"human_approval": "approved", "pending_human_input": False}
    elif human_input in ["rejected", "拒绝"]:
        return {"human_approval": "rejected", "pending_human_input": False}
    elif human_input.startswith(("modified", "修改")):
        new_requirement = human_input.replace("modified", "").replace("修改", "").strip()
        return {
            "human_approval": "modified",
            "messages": [HumanMessage(content=new_requirement)],
            "pending_human_input": False
        }
    else:
        print("指令无效，默认拒绝")
        return {"human_approval": "rejected", "pending_human_input": False}


# -------------------------- 路由函数（扩展） --------------------------
def routing_func(state: State):
    """路由函数：根据状态决定下一个节点"""
    # 1. 需要人工审核 → 跳转到人工审核节点（触发中断）
    if state.get("pending_human_input"):
        return "human_approval_node"

    # 2. 任务类型为END → 流程结束
    if state.get("type") == END:
        return END

    # 3. 根据任务类型路由到对应Worker节点
    type_map = {
        "travel": "travel_node",
        "joke": "joke_node",
        "couplet": "couplet_node",
        "other": "other_node"
    }
    return type_map.get(state["type"], "other_node")


# -------------------------- 构建图（含human-in-loop） --------------------------
# 1. 创建StateGraph实例
builder = StateGraph(State)

# 2. 添加所有节点（人工审核节点仅作为中断标记，实际逻辑在外部处理）
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("human_approval_node", lambda x: x)  # 空节点，仅用于标记中断
builder.add_node("travel_node", travel_node)
builder.add_node("joke_node", joke_node)
builder.add_node("couplet_node", couplet_node)
builder.add_node("other_node", other_node)

# 3. 定义边
builder.add_edge(START, "supervisor_node")  # 起始→监督节点

# 条件边：监督节点 → 路由到Worker/人工节点/END
builder.add_conditional_edges(
    "supervisor_node",
    routing_func,
    ["travel_node", "joke_node", "couplet_node", "other_node", "human_approval_node", END]
)

# Worker节点执行完 → 回到监督节点（触发人工审核标记）
builder.add_edge("travel_node", "supervisor_node")
builder.add_edge("joke_node", "supervisor_node")
builder.add_edge("couplet_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")

# 人工审核节点执行完 → 回到监督节点（实际逻辑在外部处理）
builder.add_edge("human_approval_node", "supervisor_node")

# 4. 配置状态存储
checkpointer = InMemorySaver()

# 5. 编译图（关键：interrupt_before 配置人工节点前中断）
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_approval_node"]  # 人工节点前触发中断
)

# -------------------------- 执行测试（核心修复：手动处理中断） --------------------------
if __name__ == "__main__":
    # 会话配置（thread_id隔离不同会话）
    config = {"configurable": {"thread_id": "human_loop_1"}}

    # 初始用户输入
    user_input = "给我讲个笑话"
    print(f"\n===== 用户初始输入：{user_input} =====")

    # 第一步：执行流程直到中断（人工审核节点前）
    initial_state = {"messages": [HumanMessage(content=user_input)], "pending_human_input": False}
    for chunk in graph.stream(initial_state, config=config, stream_mode="custom"):
        if chunk:
            print(f"【事件流】{chunk}")

    # 第二步：检查是否触发人工审核中断
    current_state = graph.get_state(config)
    if current_state.next == ["human_approval_node"]:  # 下一个节点是人工审核节点
        # 处理人工审核
        approval_update = handle_human_approval(current_state.values)
        # 更新状态并继续执行
        graph.update_state(
            config=config,
            values=approval_update,
            as_node="human_approval_node"  # 模拟人工审核节点执行完成
        )

        # 第三步：恢复流程执行直到结束
        print("\n===== 人工审核完成，继续执行流程 =====")
        for chunk in graph.stream(None, config=config, stream_mode="custom"):
            if chunk:
                print(f"【事件流】{chunk}")

    # 打印最终对话结果
    final_state = graph.get_state(config)
    print("\n===== 最终对话结果 ======")
    for msg in final_state.values["messages"]:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}: {msg.content}")