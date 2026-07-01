# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Context-scoped attributes for GenAI instrumentation.

Follows the API shape proposed by OTel spec PR #4931 (Context-scoped Attributes).
Attributes stored here are process-local — they are never serialised into W3C
Baggage headers or any outbound propagation format.

Currently used to propagate ``gen_ai.conversation_root`` from a root
WorkflowInvocation or AgentInvocation to child spans so that the root can be
identified without relying on OTel span parentage (which includes non-GenAI
parents such as HTTP spans).
"""

from __future__ import annotations

from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.context import Context

# Private context key — never leaked outside this module.
_GENAI_CONTEXT_ATTRS_KEY = otel_context.create_key(
    "opentelemetry.util.genai.context_scoped_attrs"
)


def set_context_scoped_attributes(
    attrs: dict[str, Any],
    context: Context | None = None,
) -> Context:
    """Return a new Context with *attrs* merged in (existing keys win).

    Keys already present in the context are **not** overwritten — lower-priority
    semantics matching the CSA spec: the first writer (outermost scope) wins.

    Args:
        attrs: Attributes to add to the context.
        context: Base context to merge into. Defaults to the current context.

    Returns:
        A new Context containing the merged attributes. The caller is
        responsible for attaching it if needed.
    """
    ctx = context if context is not None else otel_context.get_current()
    existing: dict[str, Any] = (
        otel_context.get_value(_GENAI_CONTEXT_ATTRS_KEY, context=ctx) or {}
    )
    # Existing keys win — new attrs only fill in gaps.
    merged = {**attrs, **existing}
    return otel_context.set_value(_GENAI_CONTEXT_ATTRS_KEY, merged, ctx)


def get_context_scoped_attributes(
    context: Context | None = None,
) -> dict[str, Any]:
    """Return context-scoped GenAI attributes, or an empty dict.

    Args:
        context: Context to read from. Defaults to the current context.

    Returns:
        A dict of attributes previously set via
        :func:`set_context_scoped_attributes`, or ``{}`` if none are present.
    """
    ctx = context if context is not None else otel_context.get_current()
    return otel_context.get_value(_GENAI_CONTEXT_ATTRS_KEY, context=ctx) or {}
