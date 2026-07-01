# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from opentelemetry._logs import Logger
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.util.genai._invocation import Error, GenAIInvocation
from opentelemetry.util.genai.context_attributes import (
    get_context_scoped_attributes,
    set_context_scoped_attributes,
)
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.metrics import InvocationMetricsRecorder
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
)
from opentelemetry.util.genai.utils import (
    gen_ai_json_dumps,
    should_capture_content_on_spans,
)


class WorkflowInvocation(GenAIInvocation):
    """
    Represents a predetermined sequence of operations (e.g. agent, LLM, tool,
    and retrieval invocations). A workflow groups multiple operations together,
    accepting input(s) and producing final output(s).

    Use handler.workflow(name) rather than constructing this directly.
    """

    def __init__(
        self,
        tracer: Tracer,
        metrics_recorder: InvocationMetricsRecorder,
        logger: Logger,
        completion_hook: CompletionHook,
        name: str | None,
    ) -> None:
        """Use handler.workflow(name) rather than calling this directly."""
        _operation_name = "invoke_workflow"
        super().__init__(
            tracer,
            metrics_recorder,
            logger,
            completion_hook,
            operation_name=_operation_name,
            span_name=f"{_operation_name} {name}" if name else _operation_name,
            span_kind=SpanKind.INTERNAL,
        )
        self.name = name
        self.input_messages: list[InputMessage] = []
        self.output_messages: list[OutputMessage] = []

        # Use the context-scoped attribute key to determine whether a GenAI
        # root already exists in the current context. This correctly ignores
        # non-GenAI parents (HTTP, gRPC, etc.) — unlike checking OTel span
        # parentage directly.
        csa = get_context_scoped_attributes()
        if "gen_ai.conversation_root" not in csa:
            # No enclosing GenAI root — this invocation is the root.
            self.conversation_root = True

        # Propagate the marker to child spans via context-scoped attributes,
        # so any nested WorkflowInvocation or AgentInvocation sees it and
        # does NOT mark itself as root.
        extra_ctx = set_context_scoped_attributes(
            {"gen_ai.conversation_root": True}
        )
        self._start(self._get_base_attributes(), extra_context=extra_ctx)

    def _get_base_attributes(self) -> dict[str, Any]:
        """Return sampling-relevant attributes available at span creation time."""
        attrs: dict[str, Any] = {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name,
        }
        return attrs

    def _get_messages_for_span(self) -> dict[str, Any]:
        if not should_capture_content_on_spans():
            return {}
        optional_attrs = (
            (
                GenAI.GEN_AI_INPUT_MESSAGES,
                gen_ai_json_dumps([asdict(m) for m in self.input_messages])
                if self.input_messages
                else None,
            ),
            (
                GenAI.GEN_AI_OUTPUT_MESSAGES,
                gen_ai_json_dumps([asdict(m) for m in self.output_messages])
                if self.output_messages
                else None,
            ),
        )
        return {
            key: value for key, value in optional_attrs if value is not None
        }

    def _apply_finish(self, error: Error | None = None) -> None:
        attributes: dict[str, Any] = {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name
        }
        attributes.update(self._get_messages_for_span())
        if error is not None:
            self._apply_error_attributes(error)
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)
        self._call_completion_hook(
            inputs=self.input_messages,
            outputs=self.output_messages,
        )
        # TODO: Add workflow metrics when supported
