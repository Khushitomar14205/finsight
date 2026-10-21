from __future__ import annotations
from typing import TypedDict, Any

class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]
    plan: dict[str, Any]
    tool_name: str
    tool_result: dict[str, Any]
    answer: str
    error: str
