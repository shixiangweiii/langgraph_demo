"""最小可运行的 LangGraph 示例：State → 节点 → END，不依赖 LLM"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


# 1. 定义状态（State）
class State(TypedDict):
    message: str


# 2. 定义节点函数：返回的 dict 会合并进 State
def say_hello(state: State) -> dict:
    print("Hello, LangGraph!")
    return {"message": "Hello, LangGraph!"}


# 3. 构建图
builder = StateGraph(State)

# 添加节点
builder.add_node("hello_node", say_hello)

# 设置入口：从 START 连一条边到第一个节点（取代旧的 set_entry_point）
builder.add_edge(START, "hello_node")

# 添加结束边
builder.add_edge("hello_node", END)

# 编译图
graph = builder.compile()

# 4. 运行图
if __name__ == "__main__":
    # 初始状态（可以为空）
    initial_state = {"message": ""}
    # version="v2" 时 invoke 返回 GraphOutput：最终状态在 .value，中断在 .interrupts
    result = graph.invoke(initial_state, version="v2")
    print("Final state:", result.value)
