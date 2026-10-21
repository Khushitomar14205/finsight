"""LangGraph orchestration: route -> execute tool -> respond.

- With an LLM configured, the LLM proposes a plan (tool + arguments as JSON);
  the plan is validated against TOOL_REGISTRY before anything runs, and the
  numbers in the answer come only from the tool's output.
- Without a key (or on any LLM failure), a deterministic rule-based router
  does the same job. The agent never executes LLM-generated code.
"""
from __future__ import annotations
import json, re
from .state import AgentState
from .tools import TOOL_REGISTRY, AgentContext
from .prompts import SYSTEM_PROMPT, PLAN_PROMPT, SUMMARY_PROMPT
from .humanize import humanize

_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_YEAR = re.compile(r"\b(20\d{2})\b")

def extract_symbols(question: str, available: list[str]) -> list[str]:
    q = " " + re.sub(r"[^A-Z0-9]", " ", question.upper()) + " "
    return [s for s in available if f" {s} " in q]

def extract_dates(question: str) -> tuple[str | None, str | None]:
    dates = _DATE.findall(question)
    if len(dates) >= 2: return dates[0], dates[1]
    if len(dates) == 1: return dates[0], None
    years = _YEAR.findall(question)
    if years:
        y = years[0]
        return f"{y}-01-01", f"{y}-12-31"
    return None, None

def rule_based_plan(question: str, ctx: AgentContext) -> dict:
    q = question.lower()
    symbols = extract_symbols(question, ctx.store.available_symbols())
    start, end = extract_dates(question)
    plan = {"symbols": symbols, "start": start, "end": end}
    if "correlation" in q or "covariance" in q:
        plan["tool"] = "correlation_covariance"
    elif "what if" in q or "what happens" in q:
        plan["tool"] = "what_if_analysis"
    elif "optimi" in q or "minimum variance" in q or "maximum sharpe" in q or "max sharpe" in q:
        plan["tool"] = "portfolio_optimization"
        plan["method"] = "maximum_sharpe" if "sharpe" in q else "minimum_variance"
    elif "machine learning" in q or q.strip().startswith("ml") or "forecast" in q or "experiment" in q:
        plan["tool"] = "run_ml_experiment"
    elif "quality" in q:
        plan["tool"] = "data_quality_report"
    elif "report" in q:
        plan["tool"] = "generate_report"
    elif "indicator" in q or "rsi" in q or "macd" in q or "moving average" in q or "bollinger" in q:
        plan["tool"] = "calculate_indicators"
    elif "compare" in q or " vs " in q or " versus " in q:
        plan["tool"] = "calculate_volatility_drawdown" if "volatilit" in q else "compare_stocks"
        if "volatilit" in q and len(symbols) >= 2: plan["tool"] = "compare_stocks"
    elif "portfolio" in q and any(k in q for k in ("risk", "sharpe", "drawdown", "volatilit", "diversif")):
        plan["tool"] = "portfolio_risk"
    elif "portfolio" in q:
        plan["tool"] = "portfolio_performance"
    elif "drawdown" in q or "volatilit" in q or "sharpe" in q:
        plan["tool"] = "calculate_volatility_drawdown"
    elif "available" in q or "data for" in q or "coverage" in q:
        plan["tool"] = "check_data_availability"
    elif "price" in q:
        plan["tool"] = "get_historical_prices"
    elif "return" in q or "gain" in q or "performance" in q:
        plan["tool"] = "calculate_returns"
    else:
        plan["tool"] = "check_data_availability"
    # Portfolio intents default to the saved portfolio symbols when none named
    if plan["tool"] in ("portfolio_risk", "portfolio_performance", "portfolio_optimization", "correlation_covariance") and not symbols:
        plan["symbols"] = list(ctx.portfolio_weights.keys())
    return plan

def llm_plan(question: str, ctx: AgentContext, settings) -> dict | None:
    """Ask the configured LLM for a routing plan. Returns None on any failure."""
    try:
        if settings.llm_provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=settings.llm_model, google_api_key=settings.gemini_api_key, temperature=0)
        elif settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)
        else:
            return None
        msg = llm.invoke(SYSTEM_PROMPT + "\n\n" + PLAN_PROMPT.format(
            question=question, symbols=", ".join(ctx.store.available_symbols()[:200])))
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m: return None
        plan = json.loads(m.group(0))
        if plan.get("tool") not in TOOL_REGISTRY: return None
        return plan
    except Exception:
        return None

def _make_llm(settings):
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.llm_model, google_api_key=settings.gemini_api_key, temperature=0)
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)
    return None

def llm_summarize(question: str, result: dict, settings) -> str | None:
    """Ask the LLM to phrase the tool result as a clean, human answer.
    Returns None on any failure so the deterministic humanizer takes over."""
    try:
        llm = _make_llm(settings)
        if llm is None: return None
        msg = llm.invoke(SYSTEM_PROMPT + "\n\n" + SUMMARY_PROMPT.format(
            question=question, result=json.dumps(result, default=str)[:6000]))
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        text = text.strip()
        # Guardrails: reject anything that smells like a raw dump or an error.
        if not text or len(text) < 10 or text.startswith("{") or "tool_used" in text:
            return None
        return text
    except Exception:
        return None

def _extreme_move_note(tool_name: str, result: dict) -> str:
    """Unadjusted prices turn splits/bonuses into fake crashes (e.g. RELIANCE's
    1:1 bonus on 2024-10-28 reads as a -49.8% day). Flag it in plain words."""
    if tool_name == "calculate_returns":
        wd = result.get("worst_day") or {}
        if isinstance(wd.get("return"), (int, float)) and wd["return"] <= -0.25:
            return (f"\n\n⚠️ Note: this window contains an extreme single-day move "
                    f"({wd['return']:+.1%} on {wd.get('date')}). Prices here are unadjusted, so this "
                    f"usually reflects a stock split or bonus issue rather than an actual loss of that size.")
    return ""

def format_answer(tool_name: str, result: dict, plan: dict, question: str = "", settings=None) -> str:
    """The user-facing answer: LLM phrasing when available, otherwise the
    deterministic plain-English humanizer. Raw tool JSON never reaches the
    chat — it stays available in the result's tool trace."""
    if tool_name == "generate_report":
        return result.get("report_markdown", "Report generated.")
    note = _extreme_move_note(tool_name, result)
    if settings is not None and getattr(settings, "has_llm", False):
        summary = llm_summarize(question, result, settings)
        if summary:
            return summary + note
    return humanize(tool_name, result, plan) + note

def _route(state: AgentState, ctx: AgentContext, settings=None) -> AgentState:
    plan = None
    if settings is not None and getattr(settings, "has_llm", False):
        plan = llm_plan(state["question"], ctx, settings)
    if plan is None:
        plan = rule_based_plan(state["question"], ctx)
    return {**state, "plan": plan, "tool_name": plan["tool"]}

def _execute(state: AgentState, ctx: AgentContext) -> AgentState:
    plan = state["plan"]
    fn = TOOL_REGISTRY[state["tool_name"]]
    kwargs = {k: v for k, v in plan.items() if k != "tool" and v is not None}
    try:
        result = fn(ctx, **kwargs)
        return {**state, "tool_result": result}
    except Exception as e:
        return {**state, "error": str(e)}

def _respond(state: AgentState, settings=None) -> AgentState:
    if state.get("error"):
        answer = (f"I couldn't complete that with the available data/tools.\n\n"
                  f"Reason: {state['error']}\n\n"
                  "Nothing was fabricated — please check the symbol names, date range, "
                  "and whether the data has been ingested (Data Quality page) or a portfolio has been built.")
        return {**state, "answer": answer}
    return {**state, "answer": format_answer(state["tool_name"], state["tool_result"], state["plan"],
                                              question=state.get("question", ""), settings=settings)}

def build_graph(ctx: AgentContext, settings=None):
    """Compile the LangGraph workflow; falls back to the same nodes run
    directly if langgraph is unavailable."""
    try:
        from langgraph.graph import StateGraph, START, END
        g = StateGraph(AgentState)
        g.add_node("route", lambda s: _route(s, ctx, settings))
        g.add_node("execute", lambda s: _execute(s, ctx))
        g.add_node("respond", lambda s: _respond(s, settings))
        g.add_edge(START, "route"); g.add_edge("route", "execute")
        g.add_edge("execute", "respond"); g.add_edge("respond", END)
        return g.compile()
    except Exception:
        return None

def run_agent(question: str, ctx: AgentContext, settings=None, history: list | None = None) -> dict:
    state: AgentState = {"question": question, "history": history or []}
    graph = build_graph(ctx, settings)
    if graph is not None:
        final = graph.invoke(state)
    else:
        final = _respond(_execute(_route(state, ctx, settings), ctx), settings)
    return {"answer": final.get("answer", ""), "tool_used": final.get("tool_name"),
            "plan": final.get("plan"), "tool_result": final.get("tool_result"),
            "error": final.get("error")}
