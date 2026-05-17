"""
Parallel / map-reduce agent built with StateGraph and the Send API.

Pattern:
  1. A *splitter* node breaks the user request into sub-tasks.
  2. The Send API fans out — each sub-task is processed by a *worker* node
     running in parallel.
  3. A *reducer* node collects all results and asks the LLM to synthesise a
     final answer.

OpenTelemetry LangChain instrumentation traces every LLM call.
"""

from typing import Annotated, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
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


# --- State definitions ----------------------------------------------------

class OverallState(TypedDict):
    topic: str
    subtopics: list[str]
    results: Annotated[list[str], lambda a, b: a + b]  # reducer: accumulate
    final_answer: str


class WorkerState(TypedDict):
    topic: str
    subtopic: str
    results: Annotated[list[str], lambda a, b: a + b]


# --- Graph nodes ----------------------------------------------------------

def splitter(state: OverallState) -> dict:
    """Hardcoded sub-tasks for the demo; in production ask the LLM to split."""
    subtopics = [
        f"{state['topic']} – history",
        f"{state['topic']} – current applications",
        f"{state['topic']} – future outlook",
    ]
    return {"subtopics": subtopics}


def fan_out(state: OverallState) -> list[Send]:
    """Return one Send per subtopic to run workers in parallel."""
    return [
        Send("worker", {"topic": state["topic"], "subtopic": st, "results": []})
        for st in state["subtopics"]
    ]


def worker(state: WorkerState, llm: ChatOpenAI) -> dict:
    """Research a single subtopic and return a short summary."""
    response = llm.invoke(
        [
            SystemMessage(content="You are a concise research assistant."),
            HumanMessage(
                content=f"Write 1–2 sentences about: {state['subtopic']}"
            ),
        ]
    )
    return {"results": [f"• {state['subtopic']}: {response.content}"]}


def reducer(state: OverallState, llm: ChatOpenAI) -> dict:
    """Synthesise the parallel results into a final answer."""
    combined = "\n".join(state["results"])
    response = llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant that summarises research."),
            HumanMessage(
                content=f"Combine these notes into a coherent 3-sentence summary:\n{combined}"
            ),
        ]
    )
    return {"final_answer": response.content}


# --- Build graph ----------------------------------------------------------

def build_graph(llm: ChatOpenAI):
    # Bind the llm into worker/reducer via closures
    def _worker(state: WorkerState) -> dict:
        return worker(state, llm)

    def _reducer(state: OverallState) -> dict:
        return reducer(state, llm)

    builder = StateGraph(OverallState)
    builder.add_node("splitter", splitter)
    builder.add_node("worker", _worker)
    builder.add_node("reducer", _reducer)

    builder.add_edge(START, "splitter")
    builder.add_conditional_edges("splitter", fan_out, ["worker"])
    builder.add_edge("worker", "reducer")
    builder.add_edge("reducer", END)

    return builder.compile()


def main():
    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, seed=100)
    graph = build_graph(llm)

    result = graph.invoke({"topic": "Artificial Intelligence", "subtopics": [], "results": [], "final_answer": ""})

    print("Parallel map-reduce agent output:")
    print("\nIndividual results:")
    for r in result["results"]:
        print(f"  {r}")
    print(f"\nFinal answer:\n  {result['final_answer']}")

    LangChainInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
