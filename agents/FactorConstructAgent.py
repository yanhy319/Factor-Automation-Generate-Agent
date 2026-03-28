from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Union
from utils.interpreter import execute_node, parse_expression_to_node
from utils.error_utils import build_retry_feedback, record_error_event


def _project_root() -> Path:
    # agents/FactorConstrcutAgent.py -> FactorGenAgent/
    return Path(__file__).resolve().parents[1]


def _resolve_parquet_path(parquet_path: str) -> Path:
    # Support the user-provided "/data/stock_data.parquet" style.
    # Treat absolute-like paths starting with "/" as relative to project root's folder.
    if parquet_path.startswith("/"):
        return _project_root() / parquet_path.lstrip("/")
    return Path(parquet_path)


class FCA:
    """
    Factor Construction / Execution Agent.

    Responsibilities:
    - Accept KEA instruction JSON (factor expressions)
    - Use utils/interpreter.py to parse expression -> node
    - Execute node on loaded stock panel data to get factor wide table
    """

    def __init__(self, parquet_path: str = "/data/stock_data.parquet"):
        self.parquet_path = parquet_path
        self.factor_names = []

    def handle_instruction(self, instruction):
        resolved_path = _resolve_parquet_path(self.parquet_path)
        df_panel = pd.read_parquet(resolved_path)

        # Normalize to a list of factor objects.
        if isinstance(instruction, dict) and instruction.get("no_factor") is True:
            return {"no_factor": True}

        factors: List[Dict[str, Any]] = []
        if isinstance(instruction, dict):
            factors = [instruction]
        elif isinstance(instruction, list):
            factors = instruction
        else:
            raise ValueError("instruction must be a dict, a list of dicts, or {'no_factor': true}")

        df_factors: List[pd.DataFrame] = []
        for i, item in enumerate(factors):
            if not isinstance(item, dict):
                raise ValueError(f"Each factor item must be a dict, got {type(item).__name__} at index {i}")

            expression = item.get("expression")
            factor_name = item.get("factor_name", f"factor_{i}")

            if not isinstance(expression, str) or not expression.strip():
                raise ValueError(f"Missing or empty 'expression' for factor at index {i}")

            node_or_err = parse_expression_to_node(expression)
            # parse_expression_to_node returns either node dict or ParseError dataclass
            if not isinstance(node_or_err, dict):
                errors = getattr(node_or_err, "errors", None)
                err_msg = f"parse_expression_to_node failed for expression='{expression}'. errors={errors}"
                event = record_error_event(
                    stage="fca.handle_instruction.parse_expression",
                    error=err_msg,
                    current_output=expression,
                    extra={"factor_name": factor_name, "index": i},
                )
                return {"ok": False, "error": err_msg, "feedback": build_retry_feedback(event)}

            try:
                df_factor = execute_node(node_or_err, df_panel)
            except Exception as e:
                err_msg = f"execute_node failed for expression='{expression}': {e}"
                event = record_error_event(
                    stage="fca.handle_instruction.execute_node",
                    error=err_msg,
                    current_output=expression,
                    extra={"factor_name": factor_name, "index": i},
                )
                return {"ok": False, "error": err_msg, "feedback": build_retry_feedback(event)}
            df_factors.append(df_factor.stack().to_frame(factor_name))
            self.factor_names.append(factor_name)

        return {"ok": True, "df_factors": df_factors}

    def backtest(self, factor_name, df_factor):
        """
        Compute per-date IC sequence only (no extra aggregation).

        Inputs:
          - df_factor:
              - wide: index=Trddt, columns=Stkcd
              - OR long panel: index=(Trddt, Stkcd), columns=factor_name(s)

        Output:
          - if single factor: pd.Series indexed by Trddt, value=IC(Trddt)
          - if multiple factors in long panel: pd.DataFrame indexed by Trddt, columns=factor_name
        """

        resolved_path = _resolve_parquet_path(self.parquet_path)
        df_panel = pd.read_parquet(resolved_path)
        ret_wide = df_panel.pivot(index="Trddt", columns="Stkcd", values="ChangeRatio").sort_index()
        # Align: factor(t) with ret(t+1) -> ret_next = ret.shift(-1) along the Trddt axis.
        ret_wide = ret_wide.shift(-1)

        df_factor = df_factor.reset_index()
        df_factor = df_factor.pivot(index="Trddt", columns="Stkcd", values=factor_name).sort_index()

        # Align by Trddt and Stkcd columns.
        common_dates = df_factor.index.intersection(ret_wide.index)

        ic_values = []
        for d in common_dates:
            f = df_factor.loc[d]
            r = ret_wide.loc[d]

            # Align on Stkcd intersection implicitly via mask.
            mask = f.notna() & r.notna()
            if mask.sum() < 2:
                ic_values.append(float("nan"))
                continue

            ic_values.append(f[mask].corr(r[mask]))

        return pd.Series(ic_values, index=common_dates, name=f'{factor_name}_ic')

