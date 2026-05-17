"""
Human-in-the-loop agent built with StateGraph.

Before executing a tool call the graph pauses and asks the user for approval.
Uses LangGraph's `interrupt` function and `InMemorySaver` checkpointer so
graph state is persisted between the first run (pause) and the resumed run
(after human decision).

Flow:
  START -> agent -> approval_gate -> tools -> agent -> ... -> END

The approval_gate node calls `interrupt()` to surface the pending tool call to
the human.  When the graph is resumed the gate checks the human's decision:
  - "approve" -> forward to tools
  - anything else  -> skip tools, return a cancellation message

OpenTelemetry LangChain instrumentation traces all LLM calls.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt

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


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (mock)."""
    return f"Email sent to {to} with subject '{subject}'."


@tool
def delete_file(path: str) -> str:
    """Delete a file (mock)."""
    return f"File '{path}' deleted."


def build_graph(llm_with_tools):

    tool_node = ToolNode([send_email, delete_file])

    def call_model(state: MessagesState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def approval_gate(state: MessagesState) -> dict:
        """Pause and ask the human to approve any pending tool calls."""
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        tool_summary = ", ".join(
            f"{tc['name']}({tc['args']})" for tc in last.tool_calls
        )
        decision = interrupt(
            f"Agent wants to call: {tool_summary}\nApprove? (approve/deny): "
        )

        if decision.strip().lower() != "approve":
            # Replace the tool calls with a cancellation notice so the model
            # can respond gracefully without a dangling tool call.
            cancel_messages = [
                ToolMessage(
                    tool_call_id=tc["id"],
                    content="Action cancelled by user.",
                )
                for tc in last.tool_calls
            ]
            return {"messages": cancel_messages}

        return {}

    def gate_route(state: MessagesState):
        """After the gate, go to tools only if the last AI message still has tool calls."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "agent"

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "approval_gate", END: END},
    )
    builder.add_conditional_edges(
        "approval_gate",
        gate_route,
        {"tools": "tools", "agent": "agent"},
    )
    builder.add_edge("tools", "agent")

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


def main():
    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, seed=42)
    llm_with_tools = llm.bind_tools([send_email, delete_file])
    graph = build_graph(llm_with_tools)

    config = {"configurable": {"thread_id": "hitl-demo-1"}}
    initial_input = {
        "messages": [
            SystemMessage(content="You are a helpful assistant with access to email and file tools."),
            HumanMessage(content="Send an email to alice@example.com with subject 'Hello' and body 'Hi there!'"),
        ]
    }

    print("--- First run (will pause for approval) ---")
    for event in graph.stream(initial_input, config=config, stream_mode="updates"):
        for node, update in event.items():
            if node == "__interrupt__":
                prompt = update[0].value
                print(f"\n[INTERRUPT] {prompt}")
                human_input = input("Your decision: ").strip()

    print("\n--- Resuming with human decision ---")
    from langgraph.types import Command
    result = graph.invoke(Command(resume=human_input), config=config)

    print("\nFinal conversation:")
    for msg in result["messages"]:
        print(f"  {type(msg).__name__}: {msg.content}")

    LangChainInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
