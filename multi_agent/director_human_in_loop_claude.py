"""
完整版 Human-in-the-Loop 工作流：LLM 驱动，两个人工检查点

START → classify_task → generate_plan → human_review_plan（approve） → execute_task → human_review_result（confirm） → finalize → END
"""
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_qwq import ChatQwen
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Interrupt, interrupt
from pydantic import BaseModel, model_validator

# TMD还得是claude，其他模型代码能力都弱鸡爆了


class State(TypedDict):
    """全局状态定义"""
    messages: Annotated[list[AnyMessage], add_messages]  # add_messages 是消息列表的标准 reducer：追加新消息，同 id 的消息会被替换
    task_type: str  # 任务类型
    plan: str  # 生成的计划
    human_approved: bool  # 人工审批状态
    execution_result: str  # 执行结果
    feedback: str  # 最近一次人工意见：驳回/修改/重做时写入，下一轮生成时拼进提示词


@dataclass
class Context:
    """运行时上下文（context_schema）：放 LLM 这类运行期依赖。它不会写进 checkpoint，所以每次 stream（包括恢复）都要传入"""
    llm: BaseChatModel


def _text_to_decision(data: Any, keywords: tuple[str, ...], default: str) -> Any:
    """把人工回复规整成 {"action", "feedback"}：控制台这类客户端只能回一句话，结构化客户端直接传 dict"""
    if isinstance(data, dict) and isinstance(data.get("action"), str):
        return {**data, "action": data["action"].strip().lower()}
    if not isinstance(data, str):
        return data
    text = data.strip()
    lowered = text.lower()
    # 句子里包含哪个关键字就视为该操作，都不包含则按 default 处理；只输入关键字本身时没有额外意见
    action = next((k for k in keywords if k in lowered), default)
    return {"action": action, "feedback": "" if lowered == action else text}


class PlanDecision(BaseModel):
    """计划审批的回复格式，作为 interrupt() 的 response_schema：
    客户端能从 Interrupt.response_schema 拿到它的 JSON Schema；恢复值先按它校验，再作为 interrupt() 的返回值交给节点
    """
    action: Literal["approve", "reject", "revise"]
    feedback: str = ""

    @model_validator(mode="before")
    @classmethod
    def _from_text(cls, data: Any) -> Any:
        return _text_to_decision(data, ("approve", "reject"), "revise")


class ResultDecision(BaseModel):
    """结果审核的回复格式：confirm 结束流程，其余一律 redo（可附带意见）"""
    action: Literal["confirm", "redo"]
    feedback: str = ""

    @model_validator(mode="before")
    @classmethod
    def _from_text(cls, data: Any) -> Any:
        return _text_to_decision(data, ("confirm",), "redo")


def classify_task(state: State, runtime: Runtime[Context]) -> dict:
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

    response = runtime.context.llm.invoke([
        SystemMessage(prompt),
        HumanMessage(state["messages"][-1].text),
    ])
    task_type = response.text.strip().lower()

    logging.info(f"任务分类结果: {task_type}")

    if task_type not in ["travel", "joke", "article", "other"]:
        task_type = "other"

    return {
        "task_type": task_type,
        "messages": [AIMessage(f"我理解了，这是一个关于 {task_type} 的任务")]
    }


def generate_plan(state: State, runtime: Runtime[Context]) -> dict:
    """
    生成计划节点：根据任务类型生成执行计划，产出交给第一个 Human-in-the-Loop 检查点审批
    """
    logging.info("=== 进入生成计划节点 ===")

    task_type = state["task_type"]
    user_query = state["messages"][0].text  # 原始用户问题

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

    # 被驳回/要求修改后重新生成时，把人工意见带给 LLM
    if feedback := state.get("feedback"):
        prompt += f"\n请根据以下人工修改意见调整计划：{feedback}"

    response = runtime.context.llm.invoke([HumanMessage(prompt)])
    plan = response.text

    logging.info(f"生成的计划：\n{plan}")

    return {
        "plan": plan,
        "messages": [AIMessage(f"我已经制定了以下计划：\n\n{plan}")]
    }


def human_review_plan(state: State) -> Command[Literal["execute_task", "generate_plan"]]:
    """
    人工审批节点：使用 interrupt() 暂停执行，等待人工审批计划
    """
    logging.info("=== 进入人工审批节点 ===")

    # 第一次执行到 interrupt() 时，LangGraph 把 State 和中断信息存进 checkpointer，本轮 stream 随之结束
    # 恢复时本节点会从第一行重新执行，这一次 interrupt() 直接返回按 PlanDecision 校验后的恢复值（不保存局部变量和调用栈）
    # 所以 interrupt() 之前不要放调用 LLM 之类有副作用的操作
    decision = interrupt(
        {
            "type": "plan_review",
            "plan": state["plan"],
            "message": "请审批以上计划。输入 'approve' 批准，'reject' 拒绝并重新生成，或提供修改意见"
        },
        response_schema=PlanDecision,
    )

    logging.info(f"收到人工反馈: {decision}")
    reply = decision.feedback or decision.action

    if decision.action == "approve":
        # 批准：继续执行任务，修改意见已经体现在计划里，清空
        return Command(
            goto="execute_task",
            update={
                "human_approved": True,
                "feedback": "",
                "messages": [HumanMessage(f"人工审批：已批准\n反馈：{reply}")]
            }
        )
    elif decision.action == "reject":
        # 拒绝：重新生成计划
        return Command(
            goto="generate_plan",
            update={
                "human_approved": False,
                "feedback": decision.feedback,
                "messages": [HumanMessage(f"人工审批：拒绝，需要重新生成\n反馈：{reply}")]
            }
        )
    else:
        # 提供修改意见：重新生成并考虑反馈
        return Command(
            goto="generate_plan",  # 恢复执行时，据Command对象的goto属性决定下一步执行哪个节点
            update={
                "human_approved": False,
                "feedback": decision.feedback,
                "messages": [
                    HumanMessage(f"人工审批：需要修改\n修改意见：{reply}"),
                    HumanMessage(f"请根据以下修改意见重新生成计划：{reply}")
                ]
            }
        )


def execute_task(state: State, runtime: Runtime[Context]) -> dict:
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

    # 结果被要求重做时，把人工意见带给 LLM
    if feedback := state.get("feedback"):
        prompt += f"\n请根据以下人工反馈改进结果：{feedback}"

    response = runtime.context.llm.invoke([HumanMessage(prompt)])
    result = response.text

    logging.info(f"任务执行结果：\n{result}")

    return {
        "execution_result": result,
        "messages": [AIMessage(f"任务执行完成！结果如下：\n\n{result}")]
    }


def human_review_result(state: State) -> Command[Literal["finalize", "execute_task"]]:
    """
    人工审核结果节点：这是第二个 Human-in-the-Loop 检查点，等待人工确认最终结果
    """
    logging.info("=== 进入结果审核节点 ===")

    # 暂停等待人工审核结果
    decision = interrupt(
        {
            "type": "result_review",
            "result": state["execution_result"],
            "message": "请审核执行结果。输入 'confirm' 确认完成，'redo' 重新执行"
        },
        response_schema=ResultDecision,
    )

    logging.info(f"收到人工反馈: {decision}")
    reply = decision.feedback or decision.action

    if decision.action == "confirm":
        # 确认：结束流程
        return Command(
            goto="finalize",
            update={
                "messages": [HumanMessage(f"人工审核：已确认\n反馈：{reply}")]
            }
        )
    else:
        # 重做：带着意见重新执行任务
        return Command(
            goto="execute_task",
            update={
                "feedback": decision.feedback,
                "messages": [
                    HumanMessage(f"人工审核：需要重做\n反馈：{reply}"),
                    HumanMessage(f"请根据反馈重新执行：{reply}")
                ]
            }
        )


def finalize(state: State) -> dict:
    """
    最终化节点：标记任务完成
    """
    logging.info("=== 任务流程完成 ===")
    return {
        "messages": [AIMessage("✅ 所有流程已完成，任务结束！")]
    }


# 构建图
def create_graph() -> CompiledStateGraph:
    """创建 Human-in-the-Loop 工作流图"""
    builder = StateGraph(State, context_schema=Context)

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
    # human_review_plan 使用 Command 动态路由，去向由返回类型里的 Literal 声明，不能再加静态出边（否则两个分支都会执行）
    builder.add_edge("execute_task", "human_review_result")
    # human_review_result 同样使用 Command 动态路由
    builder.add_edge("finalize", END)

    # HITL 必须配 checkpointer：interrupt() 暂停时把 State 存进去，恢复时按 thread_id 取回
    return builder.compile(checkpointer=InMemorySaver())


def stream_until_pause(graph: CompiledStateGraph, graph_input: dict | Command, config: dict,
                       context: Context) -> tuple[Interrupt, ...]:
    """开一个新的 stream 跑到结束或下一个 interrupt()，边跑边打印节点输出；返回本轮挂起的中断（为空表示流程已结束）"""
    pending: tuple[Interrupt, ...] = ()
    # graph.stream()调用在遇到中断时会停止迭代，每次graph.stream()调用都是独立的
    # version="v2"：每个事件都是带 type/ns/data 的 StreamPart；中断出现在 data["__interrupt__"]
    for part in graph.stream(graph_input, config, context=context, stream_mode="updates", version="v2"):
        for node_name, node_output in part["data"].items():
            if node_name == "__interrupt__":
                pending = node_output
                continue
            print(f"\n📍 节点: {node_name}")
            if node_output and node_output.get("messages"):
                print(f"💬 {node_output['messages'][-1].text}\n")
    return pending


def show_interrupt(pending: Interrupt) -> None:
    """展示中断信息：value 是节点传给 interrupt() 的内容，response_schema 是节点声明的回复格式（JSON Schema）"""
    print("\n" + "=" * 60)
    print("⏸️  流程暂停，等待人工介入")
    print("=" * 60)

    interrupt_info = pending.value
    print(f"\n📋 中断类型: {interrupt_info.get('type', 'unknown')}")
    print(f"💡 提示: {interrupt_info.get('message', '')}\n")

    if interrupt_info.get('type') == 'plan_review':
        print(f"📝 生成的计划：\n{interrupt_info.get('plan', '')}\n")
    elif interrupt_info.get('type') == 'result_review':
        print(f"📄 执行结果：\n{interrupt_info.get('result', '')}\n")

    # 真正的前端可以按 response_schema 渲染表单；控制台直接输入一句话即可，会被规整成这个格式
    if pending.response_schema:
        actions = pending.response_schema["properties"]["action"]["enum"]
        print(f"🧾 回复格式: action 取 {actions} 之一，feedback 可选\n")


def run_interactive_demo(graph: CompiledStateGraph, context: Context):
    """
    运行交互式 Demo
    演示如何使用 Human-in-the-Loop
    """
    print("\n" + "=" * 60)
    print("🤖 Human-in-the-Loop Demo 启动")
    print("=" * 60 + "\n")

    # 配置 thread_id 用于会话管理：恢复时靠同一个 thread_id 找回暂停的状态
    config = {"configurable": {"thread_id": "demo_session_001"}}

    # 初始用户输入
    user_input = input("👤 请输入你的需求（例如：帮我规划一次北京3日游）：")
    print()

    # 初始化状态
    initial_state = {
        "messages": [HumanMessage(user_input)],
        "task_type": "",
        "plan": "",
        "human_approved": False,
        "execution_result": "",
        "feedback": ""
    }

    # 开始执行图
    print("🚀 开始处理...\n")

    try:
        pending = stream_until_pause(graph, initial_state, config, context)

        # 可能连续遇到多个检查点（驳回、重做后还会再次暂停），所以循环到没有待处理的中断为止
        while pending:
            show_interrupt(pending[0])

            # 获取人工输入
            human_input = input("👤 请输入你的决定：")

            # 继续执行，传入人工反馈
            print("\n🔄 继续执行...\n")

            # 再开一个新的 stream：同一个 config 把它接回暂停处，resume 的值成为 interrupt() 的返回值
            pending = stream_until_pause(graph, Command(resume=human_input), config, context)

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


def main():
    # 检查环境变量：LLM 在确认有 key 之后才创建，避免导入模块时就因缺 key 报错
    api_key = os.getenv("LLM_SK")
    if not api_key:
        print("⚠️  警告: 未设置 LLM_SK 环境变量")
        print("请设置: export LLM_SK='your_api_key'")
        sys.exit(1)

    # 日志配置：级别为 ERROR，节点里的 logging.info 默认不输出，调试流程时调低即可；日志文件写在脚本旁边
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(__file__).with_name('hitl_demo.log'), encoding='utf-8')
        ]
    )

    # LLM 初始化，通过运行时 context 注入各节点
    # ChatQwen 走 DashScope 的 OpenAI 兼容接口；它默认连国际站，这里指定国内站，与 LLM_SK 所属地域一致
    llm = ChatQwen(model="qwen3.7-max", api_key=api_key, api_base="https://dashscope.aliyuncs.com/compatible-mode/v1")
    run_interactive_demo(create_graph(), Context(llm=llm))


if __name__ == "__main__":
    main()
