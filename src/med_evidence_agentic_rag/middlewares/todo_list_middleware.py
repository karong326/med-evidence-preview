"""创建复杂医学任务的 Todo 规划中间件。

本文件只把当前 Todo 提示词传给 LangChain ``TodoListMiddleware``，使 Agent 在查询计划包含多个
独立条目时可写入和更新任务清单。查询拆分仍由 rewrite 工具负责，实际检索约束由其他中间件负责。
"""

from typing import Any

from langchain.agents.middleware import TodoListMiddleware


# 根据提示词，要求 Agent：
#   顺序检索
#   → 阅读当前证据
#   → 更新 Todo
#   → 删除已被覆盖的任务
#   → 决定是否继续
def build_todo_list_middleware(*, todo_prompt: str) -> Any:
    """创建复杂医学问题的待办规划中间件。"""
    return TodoListMiddleware(system_prompt=todo_prompt)
