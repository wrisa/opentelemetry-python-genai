# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    ToolDefinition,
)

__all__ = ["_prepare_tool_definitions"]


def _get_property_value(obj: Any, property_name: str) -> Any:
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj).get(property_name)

    return getattr(obj, property_name, None)


def _prepare_tool_definitions(tools: list[Any]) -> list[ToolDefinition] | None:
    if not tools:
        return None

    definitions: list[ToolDefinition] = []
    for tool in tools:
        tool_type = _get_property_value(tool, "type")
        if tool_type == "function":
            func = _get_property_value(tool, "function")
            if func:
                func_name = _get_property_value(func, "name")
                func_description = _get_property_value(func, "description")
                definitions.append(
                    FunctionToolDefinition(
                        name=str(func_name) if func_name is not None else "",
                        description=str(func_description) if func_description is not None else None,
                        parameters=_get_property_value(func, "parameters"),
                    )
                )
    return definitions
