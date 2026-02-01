"""
简化版 Human-in-the-Loop 示例
演示核心机制，不依赖 LLM API
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt


class SimpleState(TypedDict):
    """简单状态"""
    input: str
    step1_result: str
    step2_result: str
    approved: bool


def step1(state: SimpleState) -> dict:
    """第一步：生成初步结果"""
    print("🔧 执行步骤1...")
    result = f"步骤1处理结果：已处理输入 '{state['input']}'"
    return {"step1_result": result}


def human_check(state: SimpleState) -> Command[Literal["step2", "step1"]]:
    """人工检查点：使用 interrupt() 暂停等待人工输入"""
    print("\n⏸️  暂停等待人工审批...")
    
    # interrupt() 会暂停执行，返回值是人工输入的内容
    # 使用interrupt()进行中断，中断后就会跳出本次graph迭代产生的stream
    # interrupt()函数实现暂停等待人工输入
    # LangGraph会保存当前执行上下文（包括局部变量、调用栈等），在检查点存储器中记录中断状态，执行上下文会被序列化
    # graph.stream()调用在遇到中断时会停止迭代，完全停止当前的stream
    print("\n⏸️ 执行interrupt()中断前")
    # 执行上下文恢复：会精确恢复到中断时的执行位置
    # 恢复后，从这里等暂停点继续执行直到下一个中断或结束
    # 生产级实现方案：在实际生产环境中的“客户端-服务器”模式下，如果服务端此时中断返回给客户端等待客户端提交，客户提交信息后服务端恢复中断之前的对话
    # 通过数据库/Redis方式持久化序列化后的会话上下文信息
    # 设计唯一会话ID: 关联用户和执行状态
    # 状态序列化: 将LangGraph检查点序列化存储
    # 恢复时：从Redis读取之前状态，根据客户端提交输入，继续执行
    # Redis作为中央状态存储：跨请求保持状态，Session ID关联：确保状态归属正确，可序列化存储：状态可以完整保存和恢复
    # 方式二：中断前保持状态到当前服务端A的内存，等客户端提交信息后，根据提交的信息转发到中断前到服务端A（会话亲和性：记录服务节点信息）
    # 纯内存方案技术上可行，性能更好：避免网络IO，内存访问更快，但是：
    # 状态丢失风险：服务重启时状态丢失
    # 节点发现困难：负载均衡器需要查找状态所在节点
    # 扩展性差：无法动态扩缩容
    # 容错性差：节点宕机会丢失状态
    # 总体来说采用redis等中心化方式：
    # 持久化保障：服务重启不丢失，
    # 高可用性：Redis集群支持，
    # 水平扩展：任意节点都能访问状态，
    # 运维友好：状态管理独立于计算节点
    # ⚠️和传统IM长连接保持的区别：
    # 核心区别：状态 vs 连接
    # LangGraph本质是状态变量，可以被序列化的应用状态数据（JSON/Protobuf），从存储中读取状态 → 重建执行上下文
    # 无状态计算层：任何节点都能处理请求
    # 状态与计算解耦：状态存外部存储
    # 天然支持水平扩展
    # IM长连接是通过已建立的 TCP 连接（连接绑定），TCP socket 文件描述符，不可被序列化，必须路由到持有该 socket 的进程
    # 连接绑定特定机器，必须会话亲和：后续消息必须路由到同一节点
    feedback = interrupt({
        "message": "请审批步骤1的结果",
        "result": state["step1_result"]
    })
    print("\n⏸️ 中断恢复，继续往下执行")
    # 这里interrupt返回的feedback，就是用户输入的 'approve'，就是Command(resume=...)中resume的值
    print(f"📥 收到人工反馈: {feedback}")
    
    # 根据人工反馈决定下一步
    if "approve" in str(feedback).lower():
        # 这里的"goto"定义了，中断恢复后跳转到哪里执行
        # 对应下面的 step2 函数
        return Command(goto="step2", update={"approved": True})
    else:
        return Command(goto="step1", update={"approved": False})


def step2(state: SimpleState) -> dict:
    """第二步：最终处理"""
    print("🔧 执行步骤2...")
    result = f"步骤2处理结果：基于 '{state['step1_result']}' 完成最终处理"
    return {"step2_result": result}


# 构建图
builder = StateGraph(SimpleState)
builder.add_node("step1", step1)
builder.add_node("human_check", human_check)
builder.add_node("step2", step2)

builder.add_edge(START, "step1")
builder.add_edge("step1", "human_check")
# human_check 使用 Command 动态路由
builder.add_edge("step2", END)

# 编译时必须提供 checkpointer
graph = builder.compile(checkpointer=MemorySaver())


def run_simple_demo():
    """运行简单示例"""
    print("\n" + "="*50)
    print("简化版 Human-in-the-Loop Demo")
    print("="*50 + "\n")
    
    config = {"configurable": {"thread_id": "simple_001"}}
    
    # 启动流程
    print("🚀 启动流程...\n")
    for event in graph.stream(
        {"input": "测试数据", "step1_result": "", "step2_result": "", "approved": False},
        config,
        stream_mode="updates"
    ):
        print(f"📍 事件: {event}")
    
    # 检查是否中断
    state = graph.get_state(config)
    if state.next:
        print("\n💡 流程已暂停，输入 'approve' 继续，其他内容将重新执行步骤1")
        human_input = input("👤 你的决定: ")
        
        print("\n🔄 继续执行...\n")
        # 再次开启一个新的stream，指定Command恢复到中断前的位置执行
        for event in graph.stream(
            Command(resume=human_input), # LangGraph Command和interrupt是实现 human-in-the-loop 的关键，通过Command对象实现动态路由，
                # Command(goto=..., update=...): 动态路由控制
                # 人工输入通过Command(resume=...)传回给图执行器
            config,
            stream_mode="updates"
        ):
            print(f"📍 事件: {event}")
    
    print("\n✅ 流程完成！")
    final_state = graph.get_state(config)
    print(f"📊 最终结果: {final_state.values.get('step2_result', 'N/A')}")


if __name__ == "__main__":
    run_simple_demo()
