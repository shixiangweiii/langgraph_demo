import os
from operator import add
from typing import TypedDict, Annotated, Literal

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt
import logging
# TMD还得是claude，其他模型代码能力都弱鸡爆了
# START → classify_task → generate_plan → human_review_plan（approve） → execute_task → human_review_result（confirm） → finalize → END
# 日志配置
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hitl_demo.log')
    ]
)

# LLM 初始化
llm = ChatTongyi(model="qwen3-max-preview", api_key=os.getenv("LLM_SK"))


class State(TypedDict):
    """全局状态定义"""
    messages: Annotated[list[AnyMessage], add]
    task_type: str  # 任务类型
    plan: str  # 生成的计划
    human_approved: bool  # 人工审批状态
    execution_result: str  # 执行结果


def classify_task(state: State) -> dict:
    """
    任务分类节点：使用 LLM 对用户问题进行分类
    """
    logging.info("=== 进入任务分类节点 ===")

    prompt = """你是一个专业的客服助手，负责对用户的问题进行分类。
    如果用户的问题是和旅游路线规划相关，返回 travel；
    如果用户问题是希望讲一个笑话，返回 joke；
    如果用户问题是希望写一篇文章，返回 article；
    如果是其他的问题，返回 other；
    只返回这四个选项之一，不要返回任何其他内容。
    """

    prompts = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["messages"][-1].content},
    ]

    response = llm.invoke(prompts)
    task_type = response.content.strip().lower()

    logging.info(f"任务分类结果: {task_type}")

    if task_type not in ["travel", "joke", "article", "other"]:
        task_type = "other"

    return {
        "task_type": task_type,
        "messages": [AIMessage(content=f"我理解了，这是一个关于 {task_type} 的任务")]
    }


def generate_plan(state: State) -> dict:
    """
    生成计划节点：根据任务类型生成执行计划
    这是第一个 Human-in-the-Loop 检查点
    """
    logging.info("=== 进入生成计划节点 ===")

    task_type = state["task_type"]
    user_query = state["messages"][0].content  # 原始用户问题

    if task_type == "travel":
        prompt = f"""请为以下旅游需求制定一个详细的旅游计划：
        用户需求：{user_query}

        请包含：
        1. 目的地推荐
        2. 行程安排（按天）
        3. 预算估算
        4. 注意事项
        """
    elif task_type == "article":
        prompt = f"""请为以下主题制定一个文章写作大纲：
        主题：{user_query}

        请包含：
        1. 文章标题
        2. 章节结构
        3. 每个章节的要点
        4. 预计字数
        """
    elif task_type == "joke":
        prompt = f"""请设计一个笑话的创作方案：
        需求：{user_query}

        请包含：
        1. 笑话主题
        2. 幽默点设计
        3. 目标受众
        """
    else:
        prompt = f"请为用户的问题制定处理方案：{user_query}"

    response = llm.invoke([{"role": "user", "content": prompt}])
    plan = response.content

    logging.info(f"生成的计划：\n{plan}")

    return {
        "plan": plan,
        "messages": [AIMessage(content=f"我已经制定了以下计划：\n\n{plan}")]
    }


def human_review_plan(state: State) -> Command[Literal["execute_task", "generate_plan"]]:
    """
    人工审批节点：等待人工审批计划
    使用 interrupt() 暂停执行，等待人工输入
    """
    logging.info("=== 进入人工审批节点 ===")

    # 使用 interrupt() 暂停执行，等待人工反馈
    # 返回值会作为提示信息
    # nterrupt()函数被调用时，LangGraph会，保存当前执行上下文（包括局部变量、调用栈等），在检查点存储器中记录中断状态，返回到调用者，暂停执行
    human_feedback = interrupt(
        {
            "type": "plan_review",
            "plan": state["plan"],
            "message": "请审批以上计划。输入 'approve' 批准，'reject' 拒绝并重新生成，或提供修改意见"
        }
    )

    logging.info(f"收到人工反馈: {human_feedback}")

    # 处理人工反馈
    if isinstance(human_feedback, dict):
        action = human_feedback.get("action", "").lower()
        feedback_text = human_feedback.get("feedback", "")
    else:
        # 兼容字符串输入
        action = str(human_feedback).lower()
        feedback_text = action

    if action == "approve" or "approve" in feedback_text:
        # 批准：继续执行任务
        return Command(
            goto="execute_task",
            update={
                "human_approved": True,
                "messages": [HumanMessage(content=f"人工审批：已批准\n反馈：{feedback_text}")]
            }
        )
    elif action == "reject" or "reject" in feedback_text:
        # 拒绝：重新生成计划
        return Command(
            goto="generate_plan",
            update={
                "human_approved": False,
                "messages": [HumanMessage(content=f"人工审批：拒绝，需要重新生成\n反馈：{feedback_text}")]
            }
        )
    else:
        # 提供修改意见：重新生成并考虑反馈
        return Command(
            goto="generate_plan", # 恢复执行时，据Command对象的goto属性决定下一步执行哪个节点
            update={
                "human_approved": False,
                "messages": [
                    HumanMessage(content=f"人工审批：需要修改\n修改意见：{feedback_text}"),
                    HumanMessage(content=f"请根据以下修改意见重新生成计划：{feedback_text}")
                ]
            }
        )


def execute_task(state: State) -> dict:
    """
    执行任务节点：根据批准的计划执行具体任务
    """
    logging.info("=== 进入任务执行节点 ===")

    task_type = state["task_type"]
    plan = state["plan"]

    if task_type == "travel":
        prompt = f"""根据以下旅游计划，生成一份详细的旅游攻略：
        {plan}

        请生成具体的可执行内容。
        """
    elif task_type == "article":
        prompt = f"""根据以下大纲，写一篇完整的文章：
        {plan}
        """
    elif task_type == "joke":
        prompt = f"""根据以下创作方案，讲一个笑话：
        {plan}
        """
    else:
        prompt = f"根据以下方案执行任务：{plan}"

    response = llm.invoke([{"role": "user", "content": prompt}])
    result = response.content

    logging.info(f"任务执行结果：\n{result}")

    return {
        "execution_result": result,
        "messages": [AIMessage(content=f"任务执行完成！结果如下：\n\n{result}")]
    }


def human_review_result(state: State) -> Command[Literal["finalize", "execute_task"]]:
    """
    人工审核结果节点：等待人工确认最终结果
    这是第二个 Human-in-the-Loop 检查点
    """
    logging.info("=== 进入结果审核节点 ===")

    # 暂停等待人工审核结果
    human_feedback = interrupt(
        {
            "type": "result_review",
            "result": state["execution_result"],
            "message": "请审核执行结果。输入 'confirm' 确认完成，'redo' 重新执行"
        }
    )

    logging.info(f"收到人工反馈: {human_feedback}")

    if isinstance(human_feedback, dict):
        action = human_feedback.get("action", "").lower()
        feedback_text = human_feedback.get("feedback", "")
    else:
        action = str(human_feedback).lower()
        feedback_text = action

    if action == "confirm" or "confirm" in feedback_text:
        # 确认：结束流程
        return Command(
            goto="finalize",
            update={
                "messages": [HumanMessage(content=f"人工审核：已确认\n反馈：{feedback_text}")]
            }
        )
    else:
        # 重做：重新执行任务
        return Command(
            goto="execute_task",
            update={
                "messages": [
                    HumanMessage(content=f"人工审核：需要重做\n反馈：{feedback_text}"),
                    HumanMessage(content=f"请根据反馈重新执行：{feedback_text}")
                ]
            }
        )


def finalize(state: State) -> dict:
    """
    最终化节点：标记任务完成
    """
    logging.info("=== 任务流程完成 ===")
    return {
        "messages": [AIMessage(content="✅ 所有流程已完成，任务结束！")]
    }


# 构建图
def create_graph():
    """创建 Human-in-the-Loop 工作流图"""
    builder = StateGraph(State)

    # 添加节点
    builder.add_node("classify_task", classify_task)
    builder.add_node("generate_plan", generate_plan)
    builder.add_node("human_review_plan", human_review_plan)
    builder.add_node("execute_task", execute_task)
    builder.add_node("human_review_result", human_review_result)
    builder.add_node("finalize", finalize)

    # 添加边
    builder.add_edge(START, "classify_task")
    builder.add_edge("classify_task", "generate_plan")
    builder.add_edge("generate_plan", "human_review_plan")
    # human_review_plan 使用 Command 动态路由，不需要显式添加边
    builder.add_edge("execute_task", "human_review_result")
    # human_review_result 使用 Command 动态路由
    builder.add_edge("finalize", END)

    # 使用 MemorySaver 作为 checkpointer
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    return graph


def run_interactive_demo():
    """
    运行交互式 Demo
    演示如何使用 Human-in-the-Loop
    """
    graph = create_graph()

    print("\n" + "=" * 60)
    print("🤖 Human-in-the-Loop Demo 启动")
    print("=" * 60 + "\n")

    # 配置 thread_id 用于会话管理
    config = {"configurable": {"thread_id": "demo_session_001"}}

    # 初始用户输入
    user_input = input("👤 请输入你的需求（例如：帮我规划一次北京3日游）：")
    print()

    # 初始化状态
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "task_type": "",
        "plan": "",
        "human_approved": False,
        "execution_result": ""
    }

    # 开始执行图
    print("🚀 开始处理...\n")

    try:
        # 使用 stream 方法逐步执行
        # graph.stream()调用在遇到中断时会停止迭代，完全停止当前的stream，每次graph.stream()调用都是独立的
        for event in graph.stream(initial_state, config, stream_mode="updates"):
            for node_name, node_output in event.items():
                print(f"\n📍 节点: {node_name}")
                if "messages" in node_output and node_output["messages"]:
                    latest_msg = node_output["messages"][-1]
                    print(f"💬 {latest_msg.content}\n")

        # 检查是否有中断（需要人工介入）
        state_snapshot = graph.get_state(config)

        while state_snapshot.next:  # 如果还有下一个节点要执行
            print("\n" + "=" * 60)
            print("⏸️  流程暂停，等待人工介入")
            print("=" * 60)

            # 获取中断信息
            if state_snapshot.tasks:
                task = state_snapshot.tasks[0]
                interrupt_info = task.interrupts[0].value if task.interrupts else {}

                print(f"\n📋 中断类型: {interrupt_info.get('type', 'unknown')}")
                print(f"💡 提示: {interrupt_info.get('message', '')}\n")

                if interrupt_info.get('type') == 'plan_review':
                    print(f"📝 生成的计划：\n{interrupt_info.get('plan', '')}\n")
                elif interrupt_info.get('type') == 'result_review':
                    print(f"📄 执行结果：\n{interrupt_info.get('result', '')}\n")

            # 获取人工输入
            human_input = input("👤 请输入你的决定：")

            # 继续执行，传入人工反馈
            print("\n🔄 继续执行...\n")

            # 每次graph.stream()调用都是独立的，所以这里是再次开启一个 graph.stream
            for event in graph.stream(
                    Command(resume=human_input),
                    config,
                    stream_mode="updates"
            ):
                for node_name, node_output in event.items():
                    print(f"\n📍 节点: {node_name}")
                    if "messages" in node_output and node_output["messages"]:
                        latest_msg = node_output["messages"][-1]
                        print(f"💬 {latest_msg.content}\n")

            # 更新状态快照
            state_snapshot = graph.get_state(config)

        print("\n" + "=" * 60)
        print("✅ 流程完成！")
        print("=" * 60 + "\n")

        # 显示最终状态
        final_state = graph.get_state(config)
        print("📊 最终状态:")
        print(f"  - 任务类型: {final_state.values.get('task_type', 'N/A')}")
        print(f"  - 人工审批: {final_state.values.get('human_approved', False)}")
        print(f"  - 执行结果: {'已完成' if final_state.values.get('execution_result') else '未完成'}")

    except Exception as e:
        logging.error(f"执行出错: {e}", exc_info=True)
        print(f"\n❌ 执行出错: {e}")


if __name__ == "__main__":
    # 检查环境变量
    if not os.getenv("LLM_SK"):
        print("⚠️  警告: 未设置 LLM_SK 环境变量")
        print("请设置: export LLM_SK='your_api_key'")
    else:
        run_interactive_demo()