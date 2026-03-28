import ast
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union
from utils.error_utils import record_error_event


JsonNode = Dict[str, Any]


def _project_root() -> Path:
    # utils/interpreter.py -> FactorGenAgent/
    return Path(__file__).resolve().parents[1]


def _load_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    root = _project_root()
    fields_path = root / "configs" / "fields.json"
    operators_path = root / "configs" / "operators.json"

    with open(fields_path, "r", encoding="utf-8") as f:
        fields_dict = json.load(f)
    with open(operators_path, "r", encoding="utf-8") as f:
        operators_dict = json.load(f)
    if not isinstance(fields_dict, dict) or not isinstance(operators_dict, dict):
        raise ValueError("fields.json and operators.json must both be JSON objects (dict).")
    return fields_dict, operators_dict


_FIELDS_DICT, _OPERATORS_DICT = _load_configs()
_ALLOWED_FIELDS: Set[str] = set(_FIELDS_DICT.keys())
_ALLOWED_OPS: Set[str] = set(_OPERATORS_DICT.keys())


def _infer_arity_from_signature(op: str) -> Optional[int]:
    meta = _OPERATORS_DICT.get(op)
    if not isinstance(meta, dict):
        return None
    sig = meta.get("signature")
    if not isinstance(sig, str):
        return None

    # Example: "ts_mean(x, n)" -> "x, n" -> arity=2
    l = sig.find("(")
    r = sig.rfind(")")
    if l == -1 or r == -1 or r <= l:
        return None
    inside = sig[l + 1 : r].strip()
    if inside == "":
        return 0
    return len([p for p in inside.split(",") if p.strip() != ""])


def _is_number_node(n: ast.AST) -> bool:
    return isinstance(n, ast.Constant) and isinstance(n.value, (int, float, bool)) and not isinstance(n.value, bool)


@dataclass(frozen=True)
class ParseError:
    ok: bool
    expression: str
    errors: Tuple[str, ...]


def parse_expression_to_node(expr: str) -> Union[JsonNode, ParseError]:
    """
    Parse KEA expression string into a JSON node tree.

    Expected expression style (from KEA prompt):
      - function-style only: add(x,y), ts_mean(x,n), rank(x), sub(x,y), ...
      - nested expressions allowed
    """
    if not isinstance(expr, str) or not expr.strip():
        err = ParseError(ok=False, expression=str(expr), errors=("Expression must be a non-empty string",))
        record_error_event(stage="interpreter.parse_expression_to_node", error=err.errors, current_output=expr)
        return err

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        err = ParseError(
            ok=False,
            expression=expr,
            errors=(f"SyntaxError: {e.msg} (line {e.lineno}, col {e.offset})",),
        )
        record_error_event(stage="interpreter.parse_expression_to_node", error=err.errors, current_output=expr)
        return err

    errors: list[str] = []
    node = _ast_to_node(tree.body, errors=errors)

    # Extra strictness: disallow any remaining unsafe syntax in the tree.
    for n in ast.walk(tree):
        if isinstance(n, (ast.Attribute, ast.Subscript, ast.Lambda, ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            errors.append(f"Disallowed syntax: {type(n).__name__}")
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.Await, ast.Yield, ast.YieldFrom)):
            errors.append(f"Disallowed syntax: {type(n).__name__}")

        # Comparisons should be expressed via gt(x,c) operator in this system.
        if isinstance(n, ast.Compare):
            errors.append("Direct comparisons are not allowed; use gt(x, c)")

    if errors:
        err = ParseError(ok=False, expression=expr, errors=tuple(errors))
        record_error_event(stage="interpreter.parse_expression_to_node", error=err.errors, current_output=expr)
        return err
    return node


def _ast_to_node(n: ast.AST, errors: list[str]) -> JsonNode:
    # field reference: close / ret
    if isinstance(n, ast.Name):
        if n.id not in _ALLOWED_FIELDS:
            errors.append(f"Unknown field '{n.id}'. Allowed: {sorted(_ALLOWED_FIELDS)}")
        return {"op": "field", "field": n.id}

    # numeric constant
    if isinstance(n, ast.Constant):
        if isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return {"op": "const", "value": float(n.value)}
        errors.append(f"Disallowed constant: {n.value!r}")
        return {"op": "const", "value": float("nan")}

    # function call: op(arg1, arg2, ...)
    if isinstance(n, ast.Call):
        if n.keywords:
            errors.append("Keyword arguments are not allowed in operator calls")
        if not isinstance(n.func, ast.Name):
            errors.append("Only simple function calls are allowed, e.g. ts_mean(close, 10)")
            return {"op": "invalid"}

        op = n.func.id
        if op not in _ALLOWED_OPS:
            errors.append(f"Operator '{op}' is not in allowed operators: {sorted(_ALLOWED_OPS)}")
            return {"op": "invalid"}

        expected = _infer_arity_from_signature(op)
        if expected is not None and len(n.args) != expected:
            errors.append(f"Operator '{op}' expects {expected} args, got {len(n.args)}")

        # Parse arguments based on operator family.
        if op in {"add", "sub", "mul", "div"}:
            if len(n.args) != 2:
                errors.append(f"Operator '{op}' expects 2 args")
            return {"op": op, "left": _ast_to_node(n.args[0], errors), "right": _ast_to_node(n.args[1], errors)}

        if op in {"neg", "rank", "zscore"}:
            if len(n.args) != 1:
                errors.append(f"Operator '{op}' expects 1 arg")
            return {"op": op, "x": _ast_to_node(n.args[0], errors)}

        if op in {"ts_mean", "ts_sum", "ts_prod"}:
            if len(n.args) != 2:
                errors.append(f"Operator '{op}' expects 2 args: ts_x(field_or_expr, window)")
            # window must be numeric constant (int/float)
            x_node = _ast_to_node(n.args[0], errors)
            win_node = n.args[1]
            if not isinstance(win_node, ast.Constant) or not isinstance(win_node.value, (int, float)) or isinstance(win_node.value, bool):
                errors.append(f"Second argument of '{op}' must be a numeric constant (window), got {type(win_node).__name__}")
                window = None
            else:
                # rolling window in pandas should be int
                window = int(win_node.value)
            return {"op": op, "x": x_node, "window": window}

        if op == "gt":
            if len(n.args) != 2:
                errors.append("Operator 'gt' expects 2 args: gt(x, c)")
            x_node = _ast_to_node(n.args[0], errors)
            c_node = n.args[1]
            if not isinstance(c_node, ast.Constant) or not isinstance(c_node.value, (int, float)) or isinstance(c_node.value, bool):
                errors.append(f"Second argument of 'gt' must be numeric constant, got {type(c_node).__name__}")
                c_val = float("nan")
            else:
                c_val = float(c_node.value)
            return {"op": "gt", "x": x_node, "c": c_val}

        # Fallback (shouldn't happen if configs/operators.json matches)
        errors.append(f"Operator '{op}' not handled in interpreter")
        return {"op": "invalid"}

    # optional: support infix / unary if someone generates it
    if isinstance(n, ast.BinOp):
        # In this system we prefer function-style; still map it to node for robustness.
        op_map = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
        mapped = op_map.get(type(n.op))
        if mapped is None:
            errors.append(f"Unsupported binary operator: {type(n.op).__name__}")
            return {"op": "invalid"}
        return {"op": mapped, "left": _ast_to_node(n.left, errors), "right": _ast_to_node(n.right, errors)}

    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return {"op": "neg", "x": _ast_to_node(n.operand, errors)}

    errors.append(f"Unsupported syntax node: {type(n).__name__}")
    return {"op": "invalid"}


def execute_node(node: JsonNode, df) -> Any:
    """
    Execute node tree on a panel long-table dataframe:
      df.columns = ['stkid', 'date', 'close', 'ret']
    Returns:
      pd.DataFrame (wide): index=date, columns=stkid
    """
    if not isinstance(node, dict) or "op" not in node:
        e = ValueError("node must be a dict with an 'op' key")
        record_error_event(stage="interpreter.execute_node", error=e, current_output=node)
        raise e

    # Helper: long -> wide for one field
    def to_wide(field: str) -> pd.DataFrame:
        wide = df.pivot(index="Trddt", columns="Stkcd", values=field)
        wide = wide.sort_index()
        return wide

    def eval_node(n: JsonNode) -> Any:
        op = n.get("op")

        if op == "field":
            return to_wide(n["field"])

        if op == "const":
            return n["value"]

        if op == "neg":
            x = eval_node(n["x"])
            return -x

        if op == "add":
            return eval_node(n["left"]) + eval_node(n["right"])

        if op == "sub":
            return eval_node(n["left"]) - eval_node(n["right"])

        if op == "mul":
            return eval_node(n["left"]) * eval_node(n["right"])

        if op == "div":
            return eval_node(n["left"]) / eval_node(n["right"])

        if op == "ts_mean":
            x = eval_node(n["x"])
            w = int(n["window"])
            return x.rolling(window=w, min_periods=w).mean()

        if op == "ts_sum":
            x = eval_node(n["x"])
            w = int(n["window"])
            return x.rolling(window=w, min_periods=w).sum()

        if op == "ts_prod":
            x = eval_node(n["x"])
            w = int(n["window"])
            # rolling.prod exists but behaves differently with NaNs; use apply for clarity.
            return x.rolling(window=w, min_periods=w).apply(lambda s: np.prod(s.values), raw=False)

        if op == "rank":
            x = eval_node(n["x"])
            return x.rank(axis=1, pct=True, method="average")

        if op == "zscore":
            x = eval_node(n["x"])
            mean = x.mean(axis=1, skipna=True)
            std = x.std(axis=1, ddof=0, skipna=True).replace(0.0, np.nan)
            return x.sub(mean, axis=0).div(std, axis=0)

        if op == "gt":
            x = eval_node(n["x"])
            c = float(n["c"])
            return (x > c).astype(float)

        raise ValueError(f"Unsupported node op: {op}")

    try:
        out = eval_node(node)
        return out
    except Exception as e:
        record_error_event(stage="interpreter.execute_node", error=e, current_output=node)
        raise
