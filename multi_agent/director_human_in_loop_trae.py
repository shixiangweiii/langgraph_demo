import os
from operator import add
from os import write
from typing import TypedDict, Annotated, List, Dict, Any

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.constants import START, END
from langgraph.graph import StateGraph
import logging

# 模拟 LLM 类，避免实际 API 调用
class MockLLM:
    def invoke(self, prompts: List[Dict[str, str]]) -> Any:
        # 简单的模拟逻辑，根据用户输入返回对应的分类
        user_content = prompts[-1]['content']
        
        if '笑话' in user_content:
            return type('obj', (object,), {'content': 'joke'})()
        elif '旅游' in user_content or '路线' in user_content:
            return type('obj', (object,), {'content': 'travel'})()
        elif '对联' in user_content:
            return type('obj', (object,), {'content': 'couplet'})()
        else:
            return type('obj', (object,), {'content': 'other'})()

# 必须在首次 logging 调用前配置！
logging.basicConfig(
    level=logging.ERROR,  # 设置最低记录级别
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('app.log')  # 同时写入文件（可选）
    ]
)

# LangGraph就是用代码的方式来绘制流程图
# 有向图（a->b后，b->后返回）+ 全局状态变量
# 其实就是类似之前自己改造dify ETL的pipeline的代码版，思想是一样的

# 使用模拟 LLM 避免实际 API 调用
llm = MockLLM()


class State(TypedDict):
    '''
    StateGraph class
    '''
    # 定义全局状态结构
    # 自定义状态变量，消息列表，指定字段更新策略为"add" -> 追加列表
    messages: Annotated[list[AnyMessage], add]
    # 类型记录subAgent执行的任务类型
    type: str
    # 人工确认结果
    human_confirmation: bool = False
    # 是否等待人工确认
    awaiting_human_confirmation: bool = False


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


def human_confirmation_node(state: State):
    '''
    人工确认节点
    :param state:
    :return:
    '''
    # print(">>> human_confirmation_node")
    logging.info(f"human_confirmation_node called with type={state.get('type', '')}")
    writer = get_stream_writer()
    writer({"node": "进入人工确认环节"})
    
    # 显示任务分类结果，等待人工确认
    user_question = state["messages"][-1].content
    task_type = state["type"]
    
    writer({"human_confirmation": {
        "message": f"用户问题: {user_question}",
        "task_type": f"分类结果: {task_type}",
        "prompt": "请确认此分类是否正确？"
    }})
    
    # 实际等待用户命令行交互输入
    print(f"\n=== 人工确认 ===")
    print(f"用户问题: {user_question}")
    print(f"分类结果: {task_type}")
    print("请确认此分类是否正确？ (y/n)")
    
    # 循环等待用户输入，直到输入有效的 y 或 n
    while True:
        # 这种命令行中断等待输入的方式，仅适用用这种命令行demo或者纯客户端程序
        # input() 会无限期阻塞当前线程，如果是server端环境，这会阻塞整个请求处理线程
        # input() 只能从本地终端读取，不支持远程
        # 同时在节点函数中等待用户输入：破坏了图的可预测性
        user_input = input().strip().lower()
        if user_input == 'y':
            writer({"human_confirmation_result": "人工确认：分类正确"})
            return {"human_confirmation": True, "awaiting_human_confirmation": False}
        elif user_input == 'n':
            writer({"human_confirmation_result": "人工确认：分类错误"})
            return {"human_confirmation": False, "awaiting_human_confirmation": False, "type": "other"}
        else:
            print("请输入 y 或 n")


def supervisor_node(state: State):
    '''
    supervisor节点
    :param state:
    :return:
    '''
    # print(">>> supervisor_node")
    logging.info(f"type={state.get('type', '')}, messages={state.get('messages', [])}, human_confirmation={state.get('human_confirmation', False)}")
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
    if "type" in state and "human_confirmation" in state:
        # 这里是图结束的出口
        writer({"supervisor_step": f"任务 type={state["type"]} 已执行完成，返回END"})
        return {"type": END}
    
    # 首次执行，进行任务分类
    response = llm.invoke(prompts)
    logging.info(f"llm invoke response={response}")
    type_res = response.content
    writer({"supervisor_step": f"任务规划分类结果 type={type_res}"})
    # LLM返回结果出参校验
    if type_res == "travel" or type_res == "joke" or type_res == "couplet" or type_res == "other":
        # 标记为等待人工确认
        return {"type": type_res, "awaiting_human_confirmation": True}
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
    # 检查是否需要人工确认
    if state.get("awaiting_human_confirmation", False):
        return "human_confirmation_node"
    
    # 检查人工确认结果
    if "human_confirmation" in state:
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
    
    # 初始状态，返回 other_node
    return "other_node"


builder = StateGraph(State)
# Supervisor-Worker 多智能体模式
# 添加节点
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("other_node", other_node)
builder.add_node("travel_node", travel_node)
builder.add_node("joke_node", joke_node)
builder.add_node("couplet_node", couplet_node)
builder.add_node("human_confirmation_node", human_confirmation_node)
# 添加边
builder.add_edge(START, "supervisor_node")
# 条件边，supervisor_node 出度为5
builder.add_conditional_edges("supervisor_node", routing_func,
                              ["travel_node", "joke_node", "couplet_node", "other_node", "human_confirmation_node", END])
# 人工确认节点 → supervisor_node
builder.add_edge("human_confirmation_node", "supervisor_node")
# 返回 supervisor_node 入度为4（不算start->supervisor_node）
# 子节点 → Supervisor → END，实现“执行-反馈-结束”循环
builder.add_edge("travel_node", "supervisor_node")
builder.add_edge("joke_node", "supervisor_node")
builder.add_edge("couplet_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")
# 短期记忆，检查点设置，支持多会话隔离与状态持久化
checkpointer = InMemorySaver()
# 编译构建
graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }
    # 测试场景1：人工确认分类正确
    print("=== 测试场景1：人工确认分类正确 ===")
    for chunk in graph.stream({"messages": [HumanMessage(content="给我讲个笑话")]}, config=config, stream_mode="custom"):
        print(chunk)
    
    # 测试场景2：人工确认分类错误
    print("\n=== 测试场景2：人工确认分类错误 ===")
    config2 = {
        "configurable": {
            "thread_id": "2"
        }
    }
    for chunk in graph.stream({"messages": [HumanMessage(content="给我规划一条旅游路线")]}, config=config2, stream_mode="custom"):
        print(chunk)
