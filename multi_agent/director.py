import os
from operator import add
from os import write
from typing import TypedDict, Annotated

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph
import logging

# 必须在首次 logging 调用前配置！
logging.basicConfig(
    level=logging.INFO,  # 设置最低记录级别
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('app.log')  # 同时写入文件（可选）
    ]
)

# LangGraph就是用代码的方式来绘制流程图
# 有向图（a->b后，b->后返回）+ 全局状态变量
# 其实就是类似之前自己改造dify ETL的pipeline的代码版，思想是一样的

llm = ChatTongyi(model="qwen3-max-preview", api_key=os.getenv("LLM_SK"))

# TypedDict 定义强类型的状态结构
class State(TypedDict):
    '''
    StateGraph class，全局数据容器，所有节点共享
    '''
    # 定义全局状态结构
    # 自定义状态变量，消息列表，指定字段更新策略为"add" -> 追加列表，每次更新都会追加到列表中
    messages: Annotated[list[AnyMessage], add]
    # 类型记录subAgent执行的任务类型
    type: str


# 函数式节点
# 接收 state，返回 dict（部分状态更新）
def other_node(state: State):
    '''
    兜底回复节点
    :param state: 全局状态
    :return: dict（部分状态更新）
    '''
    # print(">>> other_node")
    writer = get_stream_writer()
    # 往stream中发射chunk，写数据
    # get_stream_writer() 向 graph.stream() 发送 自定义事件（用于前端实时显示），类似chuanqing他们发AGUI事件协议包
    writer({"node": "进入兜底处理"})
    # 返回的 "messages": [HumanMessage(content="未知问题，我暂时无法处理这个任务")]，会自动merge进全局的messages状态变量
    # 返回值会 自动 merge 到全局 state 中（messages 追加，type 覆盖）。
    return {"messages": [AIMessage(content="未知问题，我暂时无法处理这个任务")], "type": "other"}


def supervisor_node(state: State):
    '''
    supervisor节点
    :param state:
    :return:
    '''
    # print(">>> supervisor_node")
    logging.info(f"type={state.get('type', '')}, messages={state.get('messages', [])}")
    writer = get_stream_writer()
    writer({"node": "开始任务规划" if not "type" in state else "检查任务完成情况"})
    # 根据用户的问题，进行问题分类，这里需要用到LLM的NLU能力
    prompt = """你是一个专业的客服助手，负责对用户的问题进行分类，并将任务分给其他Agent执行。
                如果用户的问题是和旅游路线规划相关，那就返回 travel；
                如果用户问题是希望讲一个笑话，那就返回 joke；
                如果用户问题是希望对一个对联，那就返回 couplet；
                如果是其他的问题，返回 other；
                除了这几个选项外，不要返回任何其他内容。
    """
    # ChatML格式的提示词
    prompts = [
        {"role": "system", "content": prompt},
        # 取最新用户消息 state["messages"][-1].content
        {"role": "user", "content": state["messages"][-1].content},
    ]

    # 有type表示是已经从sub_agent返回到supervisor节点，表示图执行完，返回END
    if "type" in state:
        # 这里是图结束的出口
        writer({"supervisor_step": f"任务 type={state["type"]} 已执行完成，返回END"})
        return {"type": END}
    response = llm.invoke(prompts)
    logging.info(f"llm invoke response={response}")
    type_res = response.content
    writer({"supervisor_step": f"任务规划分类结果 type={type_res}"})
    # LLM返回结果出参校验
    if type_res == "travel" or type_res == "joke" or type_res == "couplet" or type_res == "other":
        return {"type": type_res}
    else:
        raise ValueError("NLU failed, type is not in (travel, joke, couplet, other)")


def travel_node(state: State):
    # print(">>> travel_node")
    writer = get_stream_writer()
    writer({"node": "执行旅游规划任务"})
    return {"messages": [AIMessage(content="travel_node")], "type": "travel"}


def joke_node(state: State):
    # print(">>> joke_node")
    writer = get_stream_writer()
    writer({"node": "执行讲笑话任务"})
    return {"messages": [AIMessage(content="joke_node")], "type": "joke"}


def couplet_node(state: State):
    # print(">>> couplet_node")
    writer = get_stream_writer()
    writer({"node": "执行对对联任务"})
    return {"messages": [AIMessage(content="couplet_node")], "type": "couplet"}


def routing_func(state: State):
    # 根据全局state中的type判断跳转到哪个节点
    if state["type"] == "travel":
        return "travel_node"
    elif state["type"] == "joke":
        return "joke_node"
    elif state["type"] == "couplet":
        return "couplet_node"
    elif state["type"] == END:
        return END
    # 兜底是 other_node or END? -> 什么时候"END"?
    # 从产品考虑，兜底是“结束” or “这个问题我回答不了”
    return "other_node"

# StateGraph: 用于构建有状态的工作流图
builder = StateGraph(State)
# Supervisor-Worker 多智能体模式
# 添加节点
# 节点(Node): 图中的处理单元，接收状态并返回更新
# 通过节点函数的方式——接收当前状态并返回状态更新，LangGraph节点的标准模式
# 添加静态边
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("other_node", other_node)
builder.add_node("travel_node", travel_node)
builder.add_node("joke_node", joke_node)
builder.add_node("couplet_node", couplet_node)
# 添加边
# 连接节点，定义执行顺序
builder.add_edge(START, "supervisor_node")
# 条件边，supervisor_node 出度为5
builder.add_conditional_edges("supervisor_node", routing_func,
                              ["travel_node", "joke_node", "couplet_node", "other_node", END])
# 返回 supervisor_node 入度为4（不算start->supervisor_node）
# 子节点 → Supervisor → END，实现“执行-反馈-结束”循环
builder.add_edge("travel_node", "supervisor_node")
builder.add_edge("joke_node", "supervisor_node")
builder.add_edge("couplet_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")
# 短期记忆，检查点设置，支持多会话隔离与状态持久化
# 检查点(Checkpoint): 保存执行状态，支持恢复
checkpointer = InMemorySaver()
# 编译构建
graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }
    # 消费自定义事件流
    for chunk in graph.stream({"messages": [HumanMessage(content="给我讲个笑话")]}, config=config, stream_mode="custom"):
        # 这里每个chunk就是上面调用write发射的结果
        print(chunk)
