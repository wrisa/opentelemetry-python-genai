# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
LangChain chain-only example — no graph, no nodes.

Pipeline:

  question → researcher_chain → summariser_chain → final answer

Steps:
  1. *researcher_chain*  – gathers factual background on the user's question.
  2. *summariser_chain*  – condenses the researcher's output into a concise answer.

Both chains are composed with the pipe operator (|) into a single sequential
chain.  OpenTelemetry LangChain instrumentation traces both LLM calls.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from opentelemetry import _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

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


def build_chain(llm: ChatOpenAI):
    """Build a sequential LCEL chain: researcher | summariser."""

    researcher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a research assistant. Provide 2-3 factual sentences.",
            ),
            ("human", "{question}"),
        ]
    )

    summariser_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert summariser. Condense the text below into one clear sentence.",
            ),
            ("human", "{research}"),
        ]
    )

    researcher_chain = researcher_prompt | llm | StrOutputParser()

    # Bridge: wrap the researcher output so the summariser receives {"research": ...}
    def to_summariser_input(research: str) -> dict:
        return {"research": research}

    summariser_chain = summariser_prompt | llm | StrOutputParser()

    pipeline = researcher_chain | RunnableLambda(to_summariser_input) | summariser_chain

    return pipeline


def main():
    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=200,
        seed=42,
    )

    chain = build_chain(llm)

    question = "What is the capital of France?"
    print(f"Question: {question}\n")

    answer = chain.invoke({"question": question})

    print("Final summary:")
    print(f"  {answer}")

    LangChainInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
