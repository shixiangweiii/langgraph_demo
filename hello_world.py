# hello_langgraph.py

from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. 定义状态（State）
class State(TypedDict):
    message: str

# 2. 定义节点函数
def say_hello(state: State) -> dict:
    print("Hello, LangGraph!")
    return {"message": "Hello, LangGraph!"}

# 3. 构建图
builder = StateGraph(State)

# 添加节点
builder.add_node("hello_node", say_hello)

# 设置入口点
builder.set_entry_point("hello_node")

# 添加结束边
builder.add_edge("hello_node", END)

# 编译图
graph = builder.compile()

# 4. 运行图
if __name__ == "__main__":
    # 初始状态（可以为空）
    initial_state = {"message": ""}
    result = graph.invoke(initial_state)
    print("Final state:", result)