# Human-in-the-Loop (HITL) 使用指南

## 📚 核心概念

**Human-in-the-Loop** 是指在自动化流程中加入人工干预点，允许人工审批、修改或决策，确保关键步骤的质量和可控性。

## 🔑 LangGraph 1.0.7 中的 HITL 关键 API

### 1. `interrupt()` 函数
```python
from langgraph.types import interrupt

# 暂停执行，等待人工输入
human_feedback = interrupt(value)
```

**作用**：
- 暂停图的执行
- 返回值是人工通过 `Command(resume=...)` 提供的反馈

**参数**：
- `value`: 任意数据，通常包含需要人工审核的信息

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
from langgraph.checkpoint.memory import MemorySaver

# HITL 必须使用 checkpointer
graph = builder.compile(checkpointer=MemorySaver())
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

# 2. 启动执行
initial_state = {"input": "用户输入"}
for event in graph.stream(initial_state, config):
    print(event)

# 3. 检查是否中断
state_snapshot = graph.get_state(config)

# 4. 如果中断，获取中断信息
while state_snapshot.next:
    # 获取中断信息
    if state_snapshot.tasks:
        task = state_snapshot.tasks[0]
        if task.interrupts:
            interrupt_value = task.interrupts[0].value
            print(f"中断信息: {interrupt_value}")
    
    # 5. 获取人工输入
    human_input = input("请输入: ")
    
    # 6. 恢复执行
    for event in graph.stream(Command(resume=human_input), config):
        print(event)
    
    # 7. 更新状态快照
    state_snapshot = graph.get_state(config)

print("流程完成!")
```

## 🏗️ Demo 架构说明

### 完整版 Demo (human_in_loop_demo.py)

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

### 简化版 Demo (simple_hitl_demo.py)

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
```python
# 支持多种输入格式
if isinstance(feedback, dict):
    action = feedback.get("action")
else:
    action = str(feedback).lower()

# 提供默认行为
if action not in ["approve", "reject"]:
    action = "reject"  # 默认拒绝
```

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
# 设置环境变量
export LLM_SK='your_tongyi_api_key'

# 运行
python human_in_loop_demo.py
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
python simple_hitl_demo.py
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
4. **循环检查 state.next**：处理多次中断
5. **状态快照管理**：`get_state()` 获取当前状态

## 🔧 常见问题

### Q1: 为什么必须要 checkpointer?
**A**: interrupt() 需要保存中断时的状态，没有 checkpointer 无法恢复执行。

### Q2: 如何处理多个连续的人工检查点?
**A**: 使用 while 循环检查 `state.next`，每次中断后继续循环。

### Q3: 人工反馈的格式有要求吗?
**A**: 没有严格要求，可以是字符串、字典等，节点内部自行解析。

### Q4: 可以跳过中断直接执行吗?
**A**: 可以，在测试时可以预设反馈：
```python
graph.stream(initial_state, config, interrupt_before=["approval_node"])
```

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
