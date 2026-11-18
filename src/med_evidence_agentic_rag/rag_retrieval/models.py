"""定义检索阶段跨模块传递的 Pydantic 数据结构。

``Evidence`` 表示一条可展示的来源片段、分数和引用编号；``RetrievalResult`` 表示某个 query 的
命中状态及证据列表。工具、流式执行记录和前端证据面板依赖这两个稳定 JSON 模型，而不是直接
暴露 Chroma/LangChain 原始对象。
"""

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """一次知识检索返回的单条证据。"""

    # 该证据在本次回答中的引用编号，例如 "E1"；用于正文引用与证据面板对应。
    citation_id: str = Field(min_length=1)
    # 检索或重排序给出的相关性分数。
    score: float
    # 原始资料的来源标识，通常是文件路径或文件名；用于追溯证据来自哪份资料。
    source: str = Field(min_length=1)
    # 给用户展示的资料标题；缺少标题时，通常由来源文件名生成。
    title: str = Field(min_length=1)
    # 实际命中的原文片段；这是回答引用和前端展示的核心内容。
    content: str = Field(min_length=1)
    # 产生这条证据的检索问题；单问题检索时可为空；多问题检索时用于区分归属。
    query: str | None = None


class RetrievalResult(BaseModel):
    """一次知识检索的结构化结果。"""

    # 本次实际执行的检索问题；可能是用户原问题，也可能是拆分后的子问题。
    query: str = Field(min_length=1)
    # 是否至少找到一条有效证据；前端和 Agent 可据此决定是否继续使用检索结果。
    found: bool
    # 本次命中的全部证据，按相关性排序；未命中时固定为空列表，而不是 null。
    evidences: list[Evidence] = Field(default_factory=lambda: list[Evidence]())
