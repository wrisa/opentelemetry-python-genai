"""
Custom tool-calling agent built manually with StateGraph.

Equivalent in behaviour to create_react_agent but assembled from primitives:
  StateGraph + ToolNode + tools_condition

The agent loops: model -> tools -> model until the model stops calling tools.
OpenTelemetry LangChain instrumentation traces every LLM call in the loop.
"""

from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

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
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


def build_agent(llm_with_tools):
    """Build a ReAct-style agent graph manually from StateGraph primitives."""

    def call_model(state: MessagesState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode([multiply, add])

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    # Route to tools if the model made tool calls, otherwise finish.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile()


def main():
    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, seed=100)
    llm_with_tools = llm.bind_tools([multiply, add])

    agent = build_agent(llm_with_tools)

    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content="You are a helpful calculator assistant."),
                HumanMessage(content="What is (5 * 6) + 3?"),
            ]
        }
    )

    print("Custom tool-calling agent output:")
    for msg in result["messages"]:
        print(f"  {type(msg).__name__}: {msg.content}")

    LangChainInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
