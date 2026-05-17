"""
Multi-agent example built with StateGraph.

An orchestrator graph wires two specialist sub-agents together:
  - math_agent  : handles arithmetic questions using calculator tools
  - weather_agent: handles weather questions using a mock weather tool

A router node inspects the user's question and dispatches to the right agent.
OpenTelemetry LangChain instrumentation traces all LLM calls in both agents.
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain.agents import create_agent

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
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def get_weather(city: str) -> str:
    """Return a mock weather report for a city."""
    reports = {
        "london": "Cloudy, 15°C",
        "paris": "Sunny, 22°C",
        "new york": "Partly cloudy, 18°C",
    }
    return reports.get(city.lower(), f"Weather data unavailable for {city}.")


# --- Router ---------------------------------------------------------------

def route(state: MessagesState) -> str:
    """Decide which sub-agent should handle the latest user message."""
    last_content = state["messages"][-1].content.lower()
    if any(kw in last_content for kw in ("weather", "temperature", "forecast")):
        return "weather_agent"
    return "math_agent"


# --- Graph ----------------------------------------------------------------

def build_multi_agent_graph(llm: ChatOpenAI):
    from uuid import uuid4
    session_id = str(uuid4())
    math_agent = create_agent(
        llm, tools=[multiply, add], name="math_agent"
    ).with_config(
        {
            "metadata": {
                "agent_name": "math_agent",
                "session_id": session_id,
            },
        }
    )
    weather_agent = create_agent(
        llm, tools=[get_weather], name="weather_agent"
    ).with_config(
        {
            "metadata": {
                "agent_name": "weather_agent",
                "session_id": session_id,
            },
        }
    )

    def run_math_agent(state: MessagesState) -> dict:
        result = math_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"]}

    def run_weather_agent(state: MessagesState) -> dict:
        result = weather_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"]}

    builder = StateGraph(MessagesState)
    builder.add_node("math_agent", run_math_agent)
    builder.add_node("weather_agent", run_weather_agent)

    builder.add_conditional_edges(
        START,
        route,
        {"math_agent": "math_agent", "weather_agent": "weather_agent"},
    )
    builder.add_edge("math_agent", END)
    builder.add_edge("weather_agent", END)

    return builder.compile()


def main():
    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, seed=100)
    graph = build_multi_agent_graph(llm)

    questions = [
        "What is 12 multiplied by 7?",
        "What is the weather in Paris?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = graph.invoke({"messages": [HumanMessage(content=question)]})
        last = result["messages"][-1]
        print(f"  Answer: {last.content}")

    LangChainInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
