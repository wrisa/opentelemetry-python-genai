"""
Supervisor agent pattern built with StateGraph.

A supervisor LLM node decides which worker agent to call next (or when to
finish).  Workers report back to the supervisor after each turn.

Topology:
  START -> supervisor -> {researcher | writer | END}
           researcher  -> supervisor
           writer      -> supervisor

OpenTelemetry LangChain instrumentation traces all LLM calls.
"""

import json
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

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

# Workers available to the supervisor.
WORKERS = ["researcher", "writer"]

SUPERVISOR_SYSTEM = """You are a supervisor managing these workers: {workers}.
Given the conversation, decide who should act next, or reply FINISH if done.
Reply with JSON only: {{"next": "<worker>|FINISH"}}"""


def build_supervisor_graph(llm: ChatOpenAI):

    # --- Supervisor node --------------------------------------------------

    supervisor_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, seed=42)

    def supervisor(state: MessagesState) -> dict:
        prompt = SUPERVISOR_SYSTEM.format(workers=", ".join(WORKERS))
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = supervisor_llm.invoke(messages)
        return {"messages": [response]}

    def supervisor_route(
        state: MessagesState,
    ) -> Literal["researcher", "writer", "__end__"]:
        last = state["messages"][-1]
        try:
            data = json.loads(last.content)
            nxt = data.get("next", "FINISH")
        except (json.JSONDecodeError, AttributeError):
            nxt = "FINISH"
        return "__end__" if nxt == "FINISH" else nxt

    # --- Worker nodes -----------------------------------------------------

    def researcher(state: MessagesState) -> dict:
        """Mock researcher: summarises existing knowledge."""
        response = llm.invoke(
            state["messages"]
            + [SystemMessage(content="You are a researcher. Provide key facts concisely.")]
        )
        return {"messages": [AIMessage(content=f"[Researcher] {response.content}")]}

    def writer(state: MessagesState) -> dict:
        """Mock writer: drafts a short answer from the gathered facts."""
        response = llm.invoke(
            state["messages"]
            + [SystemMessage(content="You are a writer. Compose a clear, concise answer.")]
        )
        return {"messages": [AIMessage(content=f"[Writer] {response.content}")]}

    # --- Build graph ------------------------------------------------------

    builder = StateGraph(MessagesState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("researcher", researcher)
    builder.add_node("writer", writer)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        supervisor_route,
        {"researcher": "researcher", "writer": "writer", "__end__": END},
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")

    return builder.compile()


def main():
    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, seed=100)
    graph = build_supervisor_graph(llm)

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="Explain the water cycle in 2–3 sentences.")
            ]
        }
    )

    print("Supervisor agent output (full conversation):")
    for msg in result["messages"]:
        print(f"  {type(msg).__name__}: {msg.content}")

    LangChainInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
