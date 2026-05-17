"""
Plan-and-execute agent built with StateGraph.

Pattern:
  1. *Planner* node: the LLM receives the user goal and produces an ordered
     list of steps (the plan).
  2. *Executor* node: executes the current step using available tools and
     appends the result.
  3. *Replanner* node: decides whether to continue with the next step, revise
     the plan, or finish.

State machine:
  START -> planner -> executor -> replanner -> {executor | END}

OpenTelemetry LangChain instrumentation traces all LLM calls.
"""

import json
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from opentelemetry import _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

# Configure logging
_logs.set_logger_provider(LoggerProvider())
_logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)

# Configure metrics
metrics.set_meter_provider(
    MeterProvider(
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())]
    )
)


# --- Tools ----------------------------------------------------------------

@tool
def search_web(query: str) -> str:
    """Search the web and return a mock result."""
    mock_results = {
        "population of france": "France has a population of approximately 68 million people.",
        "capital of france": "The capital of France is Paris.",
        "area of france": "France covers an area of approximately 551,695 km².",
    }
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    return f"Search result for '{query}': No relevant data found."


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression safely."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: invalid characters in expression."
        return str(eval(expression))  # noqa: S307 — guarded above
    except Exception as exc:
        return f"Error: {exc}"


# --- State ----------------------------------------------------------------

class PlanExecuteState(TypedDict):
    goal: str
    plan: list[str]
    current_step_index: int
    step_results: list[str]
    final_answer: str


# --- Nodes ----------------------------------------------------------------

PLANNER_SYSTEM = """You are a planner. Given a goal, produce a JSON list of
concrete, ordered steps to achieve it. Use only simple research or calculation
steps. Respond with JSON only: {"steps": ["step1", "step2", ...]}"""

EXECUTOR_SYSTEM = """You are an executor with access to search_web and calculate tools.
Execute the given step and return the result. Be concise."""

REPLANNER_SYSTEM = """You are a replanner. Given the original goal, the plan,
and the results so far, decide:
- If all steps are done and the goal is achieved, respond: {"action": "finish", "answer": "<final answer>"}
- If the next step should proceed, respond: {"action": "continue"}
- If the plan needs revision, respond: {"action": "revise", "new_steps": ["..."]}
Respond with JSON only."""


def build_graph(llm: ChatOpenAI):
    tools = [search_web, calculate]
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    # --- Planner ----------------------------------------------------------

    def planner(state: PlanExecuteState) -> dict:
        response = llm.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content=f"Goal: {state['goal']}"),
            ]
        )
        try:
            data = json.loads(response.content)
            steps = data.get("steps", [])
        except (json.JSONDecodeError, AttributeError):
            steps = [state["goal"]]
        return {"plan": steps, "current_step_index": 0, "step_results": []}

    # --- Executor ---------------------------------------------------------

    def executor(state: PlanExecuteState) -> dict:
        idx = state["current_step_index"]
        step = state["plan"][idx]

        # Ask the model to execute this step (may use a tool).
        messages = [
            SystemMessage(content=EXECUTOR_SYSTEM),
            HumanMessage(content=f"Step: {step}"),
        ]
        response = llm_with_tools.invoke(messages)

        # If the model called a tool, run it and collect the result.
        result_text = response.content or ""
        if response.tool_calls:
            tool_messages = [response]
            for tc in response.tool_calls:
                tool_result = tool_node.invoke(
                    {"messages": tool_messages + [
                        AIMessage(content="", tool_calls=[tc])
                    ]}
                )
                last_tool_msg = tool_result["messages"][-1]
                result_text = last_tool_msg.content

        updated_results = list(state["step_results"]) + [f"Step {idx + 1}: {result_text}"]
        return {
            "step_results": updated_results,
            "current_step_index": idx + 1,
        }

    # --- Replanner --------------------------------------------------------

    def replanner(state: PlanExecuteState) -> dict:
        results_text = "\n".join(state["step_results"])
        plan_text = "\n".join(
            f"{i + 1}. {s}" for i, s in enumerate(state["plan"])
        )
        response = llm.invoke(
            [
                SystemMessage(content=REPLANNER_SYSTEM),
                HumanMessage(
                    content=(
                        f"Goal: {state['goal']}\n"
                        f"Plan:\n{plan_text}\n"
                        f"Results so far:\n{results_text}\n"
                        f"Steps completed: {state['current_step_index']} / {len(state['plan'])}"
                    )
                ),
            ]
        )
        try:
            data = json.loads(response.content)
        except (json.JSONDecodeError, AttributeError):
            data = {"action": "finish", "answer": results_text}

        if data.get("action") == "finish":
            return {"final_answer": data.get("answer", results_text)}
        if data.get("action") == "revise":
            new_steps = data.get("new_steps", [])
            return {"plan": new_steps, "current_step_index": 0}
        return {}  # continue

    def should_continue(state: PlanExecuteState) -> str:
        if state.get("final_answer"):
            return END
        if state["current_step_index"] >= len(state["plan"]):
            return "replanner"
        return "executor"

    # --- Assemble ---------------------------------------------------------

    builder = StateGraph(PlanExecuteState)
    builder.add_node("planner", planner)
    builder.add_node("executor", executor)
    builder.add_node("replanner", replanner)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges(
        "executor",
        should_continue,
        {"executor": "executor", "replanner": "replanner", END: END},
    )
    builder.add_conditional_edges(
        "replanner",
        should_continue,
        {"executor": "executor", END: END},
    )

    return builder.compile()


def main():
    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, seed=100)
    graph = build_graph(llm)

    result = graph.invoke(
        {
            "goal": "Find the population and area of France, then calculate the population density.",
            "plan": [],
            "current_step_index": 0,
            "step_results": [],
            "final_answer": "",
        }
    )

    print("Plan-and-execute agent output:")
    print("\nSteps executed:")
    for r in result["step_results"]:
        print(f"  {r}")
    print(f"\nFinal answer:\n  {result['final_answer']}")

    LangChainInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
