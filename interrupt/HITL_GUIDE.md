# Human-in-the-Loop (HITL) 使用指南

## 📚 核心概念

**Human-in-the-Loop** 是指在自动化流程中加入人工干预点，允许人工审批、修改或决策，确保关键步骤的质量和可控性。

## 🔑 LangGraph 1.2 中的 HITL 关键 API

### 1. `interrupt()` 函数
```python
from langgraph.types import interrupt

# 暂停执行，等待人工输入
human_feedback = interrupt(value)

# 1.1+ 可以声明回复格式：恢复值先按它校验，interrupt() 返回校验后的对象
decision = interrupt(value, response_schema=PlanDecision)  # Pydantic 模型 / TypedDict / dataclass
```

**作用**：
- 暂停图的执行
- 返回值是人工通过 `Command(resume=...)` 提供的反馈
- 恢复时节点会**从第一行重新执行**，这一次 `interrupt()` 直接返回恢复值；LangGraph 不保存局部变量和调用栈，所以 `interrupt()` 之前不要放调用 LLM、写库之类有副作用的操作

**参数**：
- `value`: 任意数据，通常包含需要人工审核的信息
- `response_schema`（可选）: 回复格式。客户端从 `Interrupt.response_schema` 拿到对应的 JSON Schema；恢复值校验失败会从 `graph.stream()` 抛出 `pydantic.ValidationError`，线程仍停在原中断，用合法的值再恢复一次即可

### 2. `Command` 对象
```python
from langgraph.types import Command

# 恢复执行并传入人工反馈
Command(resume=human_input)

# 或者跳转到指定节点
Command(goto="node_name", update={...})
```

**作用**：
- 恢复暂停的执行
- 动态路由到指定节点
- 更新状态

### 3. Checkpointer（必需）
```python
from langgraph.checkpoint.memory import InMemorySaver  # MemorySaver 是它的旧别名

# HITL 必须使用 checkpointer
graph = builder.compile(checkpointer=InMemorySaver())
```

**为什么必需**：
- 保存中断时的状态
- 支持恢复执行
- 实现会话持久化

## 🎯 实现模式

### 模式 1：简单审批（批准/拒绝）
```python
def approval_node(state: State) -> Command[Literal["next", "retry"]]:
    feedback = interrupt({"message": "请审批", "data": state["data"]})
    
    if "approve" in str(feedback).lower():
        return Command(goto="next")
    else:
        return Command(goto="retry")
```

### 模式 2：条件分支
```python
def decision_node(state: State) -> Command[Literal["path_a", "path_b", "path_c"]]:
    feedback = interrupt({"options": ["A", "B", "C"]})
    
    choice = str(feedback).upper()
    if choice == "A":
        return Command(goto="path_a")
    elif choice == "B":
        return Command(goto="path_b")
    else:
        return Command(goto="path_c")
```

### 模式 3：修改并继续
```python
def edit_node(state: State) -> Command[Literal["execute"]]:
    feedback = interrupt({
        "current": state["plan"],
        "message": "请审核或修改计划"
    })
    
    # 人工反馈可以是修改后的内容
    return Command(
        goto="execute",
        update={"plan": feedback}  # 使用人工修改的内容
    )
```

## 🔄 执行流程

### 完整执行示例
```python
# 1. 创建图
graph = create_graph()
config = {"configurable": {"thread_id": "session_001"}}


def stream_until_pause(graph_input):
    """开一个新的 stream 跑到结束或下一个 interrupt()，返回本轮挂起的中断"""
    pending = ()
    # version="v2"：每个事件都是带 type/ns/data 的 StreamPart；中断出现在 data["__interrupt__"]
    for part in graph.stream(graph_input, config, stream_mode="updates", version="v2"):
        print(part["data"])
        if "__interrupt__" in part["data"]:
            pending = part["data"]["__interrupt__"]
    return pending


# 2. 启动执行：遇到 interrupt() 时本轮 stream 就结束了
pending = stream_until_pause({"input": "用户输入"})

# 3. 驳回/重做会再次暂停，所以循环到没有待处理的中断为止
while pending:
    # 4. 获取中断信息（节点传给 interrupt() 的 value）
    print(f"中断信息: {pending[0].value}")

    # 5. 获取人工输入
    human_input = input("请输入: ")

    # 6. 恢复执行：同一个 thread_id 把新的 stream 接回暂停处
    pending = stream_until_pause(Command(resume=human_input))

print("流程完成!")
```

也可以在 stream 结束后用 `graph.get_state(config).interrupts` 读取挂起的中断；`graph.invoke(..., version="v2")` 返回 `GraphOutput`，最终状态在 `.value`、中断在 `.interrupts`。`stream()` / `invoke()` 默认仍是 `version="v1"`（普通 dict）。

## 🏗️ Demo 架构说明

### 完整版 Demo (multi_agent/director_human_in_loop_claude.py)

**流程图**：
```
START
  ↓
classify_task (任务分类)
  ↓
generate_plan (生成计划)
  ↓
human_review_plan (人工审批) ⏸️ HITL 检查点1
  ↓ approve              ↓ reject/modify
execute_task (执行任务)  → 返回 generate_plan
  ↓
human_review_result (结果审核) ⏸️ HITL 检查点2
  ↓ confirm              ↓ redo
finalize (完成)          → 返回 execute_task
  ↓
END
```

**两个 HITL 检查点**：
1. **计划审批**：审批 AI 生成的执行计划
2. **结果审核**：确认最终执行结果

### 简化版 Demo (interrupt/simple_hitl_demo.py)

**流程图**：
```
START → step1 → human_check ⏸️ → step2 → END
                     ↓ reject
                   返回 step1
```

**单个 HITL 检查点**：
- 审批步骤1的结果，决定继续或重做

## 💡 最佳实践

### 1. 清晰的中断信息
```python
interrupt({
    "type": "approval",           # 中断类型
    "message": "请审批此计划",     # 提示信息
    "data": state["plan"],        # 需要审核的数据
    "options": ["approve", "reject"]  # 可选操作
})
```

### 2. 健壮的反馈处理
用 `response_schema` 声明回复格式，把"字符串还是 dict"的解析集中到 schema 里，节点只处理校验后的对象：
```python
class PlanDecision(BaseModel):
    action: Literal["approve", "reject", "revise"]
    feedback: str = ""

    @model_validator(mode="before")
    @classmethod
    def _from_text(cls, data):
        # 兼容只能回一句话的客户端（如控制台）：把字符串规整成 {"action", "feedback"}
        if isinstance(data, str):
            lowered = data.strip().lower()
            action = "approve" if "approve" in lowered else "reject" if "reject" in lowered else "revise"
            return {"action": action, "feedback": data.strip()}
        return data


decision = interrupt({"message": "请审批"}, response_schema=PlanDecision)
if decision.action == "approve":
    ...
```
字符串和合法的 dict 都能用；非法的 dict（如 `{"action": "maybe"}`）会被校验拦下，线程停在原中断等待重新恢复。

### 3. 状态更新策略
```python
# 好的做法：明确更新状态
return Command(
    goto="next_node",
    update={
        "approved": True,
        "messages": [HumanMessage(content=f"审批意见: {feedback}")]
    }
)

# 避免：状态不一致
# 忘记更新关键状态变量
```

### 4. 日志和追踪
```python
def approval_node(state: State):
    logging.info(f"等待审批: {state['plan']}")
    feedback = interrupt(...)
    logging.info(f"收到反馈: {feedback}")
    # ...
```

## 🚀 运行 Demo

### 运行完整版
```bash
# 设置环境变量（DashScope / 通义千问的 API key）
export LLM_SK='your_dashscope_api_key'

# 在仓库根目录运行
.venv2/bin/python multi_agent/director_human_in_loop_claude.py
```

**交互流程**：
1. 输入需求（如：帮我规划北京3日游）
2. AI 生成计划
3. **[人工审批]** 输入 `approve` 或修改意见
4. AI 执行任务
5. **[人工审核]** 输入 `confirm` 或 `redo`
6. 完成

### 运行简化版
```bash
.venv2/bin/python interrupt/simple_hitl_demo.py
```

**交互流程**：
1. 自动执行步骤1
2. **[人工检查]** 输入 `approve` 继续或其他重做
3. 执行步骤2
4. 完成

## 🎓 学习要点

1. **interrupt() 是核心**：暂停执行的关键
2. **必须使用 checkpointer**：保存中断状态
3. **Command 控制流转**：动态路由和状态更新
4. **循环处理中断**：驳回/重做后会再次暂停，要循环到没有待处理的中断
5. **恢复 = 节点重跑**：恢复时节点从第一行重新执行，`interrupt()` 之前不要有副作用
6. **状态快照管理**：`get_state()` 获取当前状态

## 🔧 常见问题

### Q1: 为什么必须要 checkpointer?
**A**: interrupt() 需要保存中断时的状态，没有 checkpointer 无法恢复执行。

### Q2: 如何处理多个连续的人工检查点?
**A**: 用 while 循环：只要本轮 stream 里出现了 `__interrupt__`（或 `graph.get_state(config).interrupts` 不为空），就获取人工输入并用新的 stream 恢复。

### Q3: 人工反馈的格式有要求吗?
**A**: 不声明 `response_schema` 时没有要求，可以是字符串、字典等，节点内部自行解析；声明了 `response_schema` 时，恢复值会先按它校验，节点拿到的是校验后的对象。

### Q4: `interrupt_before` 能跳过中断吗?
**A**: 不能。`interrupt_before=["node"]` 是静态断点：在该节点执行*之前*额外暂停一次（没有 payload），用 `graph.stream(None, config)` 继续；节点里的 `interrupt()` 之后照样会触发。测试时要预设人工回复，直接用 `Command(resume=...)` 恢复即可（本仓库的回归验证就是用管道给 `input()` 喂数据）。

## 📊 与原 Demo 的对比

| 特性 | 原 Demo | HITL Demo |
|------|---------|-----------|
| 流程控制 | 完全自动 | 人工可介入 |
| 状态管理 | 简单状态 | 需要 checkpointer |
| 错误处理 | 抛异常 | 人工可纠正 |
| 灵活性 | 固定流程 | 动态调整 |
| 适用场景 | 简单任务 | 关键决策 |

## 🎯 实际应用场景

1. **内容审核**：AI 生成内容 → 人工审核 → 发布
2. **代码审查**：AI 生成代码 → 开发者审核 → 部署
3. **客服升级**：AI 处理 → 复杂问题转人工 → 完成
4. **财务审批**：AI 分析 → 财务审批 → 执行
5. **医疗诊断**：AI 建议 → 医生确认 → 治疗

## 📝 总结

LangGraph 的 Human-in-the-Loop 机制通过 `interrupt()` 和 `Command` 实现了优雅的人机协作：
- ✅ 保持 AI 效率
- ✅ 确保关键决策质量  
- ✅ 支持动态流程调整
- ✅ 实现真正的人机协同

这是构建生产级 AI Agent 的关键能力！
