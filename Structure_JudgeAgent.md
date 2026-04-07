## Return of ja.run_ja()

result: dict
├─ ts: str
│  含义: 本次最终 judge 决策的 UTC 时间戳
├─ agent: str
│  含义: 固定为 "JudgeAgent"
├─ pdf_path: str
│  含义: 本次任务使用的 PDF 路径
├─ query: str
│  含义: 本次任务的查询/目标描述
├─ models: list[str]
│  含义: 参与 judge 的模型列表，顺序与本次运行一致
│  └─ item: str
├─ consistent: bool
│  含义: 最终是否达到一致
├─ decision: str
│  含义: 最终判决类型
│  典型值:
│  - "consistent_factor_values"
│  - "factor_count_mismatch"
│  - "factor_value_mismatch"
│  - "models_failure"
│  - "no_factor_conflict"
│  - "empty_factor_list"
│  - "all_models_returned_no_factor"
├─ judge_iteration: int
│  含义: 最终停在第几轮 judge iteration
├─ judge_feedback_used: str | None
│  含义: 本轮生成时实际使用的上轮反馈；第一轮通常为 None
├─ latest_feedback: str | None
│  含义: 最后一轮生成出的反馈文本
│  说明:
│  - 如果最终一致，通常为最后一次不一致后生成的反馈，或 None
│  - 如果最终不一致，这是下一轮继续迭代最直接可用的反馈
├─ model_reports: dict[str, ModelReport]
│  含义: 每个模型自己的结果摘要；每个模型信息只保留在这里
│  └─ <model_name>: dict
│     ├─ status: str
│     │  含义: 该模型本轮状态
│     │  典型值:
│     │  - "ok"
│     │  - "no_factor"
│     │  - "kea_failed"
│     │  - "fca_failed"
│     │  - "unknown"
│     ├─ error: str | None
│     │  含义: 如果该模型失败，这里保存错误信息；成功则为 None
│     └─ factors: list[FactorBrief]
│        含义: 该模型本轮给出的最终因子摘要
│        说明:
│        - 成功时来自执行后的 factor_results
│        - 失败时尽量回退到 instruction 中提取
│        └─ item: dict
│           ├─ factor_name: str | None
│           │  含义: 该模型给出的因子名
│           └─ expression: str | None
│              含义: 该模型给出的表达式
├─ final_factors: list[FinalFactor]
│  含义: 已被 Judge 确认为可保留的最终因子
│  说明:
│  - 完全一致时通常是全部最终因子
│  - 部分不一致时，也可能保留已确认一致的那部分
│  - 全部失败/无因子时为空列表
│  └─ item: dict
│     ├─ factor_index: int
│     │  含义: 因子在本轮结果中的位置
│     ├─ factor_name: str | None
│     │  含义: 经过多数投票/主结果选定的最终因子名
│     ├─ expression: str | None
│     │  含义: 经过多数投票/主结果选定的最终表达式
│     ├─ core_logic: str | None
│     │  含义: 因子的核心逻辑描述
│     ├─ data_source: list[Any] | None
│     │  含义: 因子依赖的数据来源描述
│     │  └─ item: Any
│     ├─ source_models: list[str]
│     │  含义: 支持该最终因子的模型列表
│     │  └─ item: str
│     ├─ expression_consensus: bool
│     │  含义: 各模型表达式是否完全一致
│     ├─ name_consensus: bool
│     │  含义: 各模型因子名是否完全一致
│     ├─ factor_value_path: str
│     │  含义: 已保存的最终因子值文件路径
│     └─ backtest_path: str | None
│        含义: 已保存的回测结果路径；未保存则为 None
├─ issue_report: None | IssueReport
│  含义: 只有“不一致”或“异常”时才有的跨模型问题摘要
│  说明:
│  - 一致结果时为 None
│  - 所有跨模型问题信息只放在这里
│
│  变体 1: None
│
│  变体 2: dict
│  ├─ reason: str
│  │  含义: 问题类型
│  │  典型值:
│  │  - "models_failure"
│  │  - "no_factor_conflict"
│  │  - "factor_count_mismatch"
│  │  - "empty_factor_list"
│  │  - "factor_value_mismatch"
│  ├─ summary: str
│  │  含义: 对问题的简短文字总结
│  └─ mismatch_factors: dict[str, FactorMismatch]   仅在 factor_value_mismatch 时出现
│     含义: 逐个 factor_index 的数值不一致详情
│     └─ <factor_index_str>: dict
│        ├─ factor_name_by_model: dict[str, str | None]
│        │  含义: 各模型在该位置上的因子名
│        │  └─ <model_name>: str | None
│        ├─ expression_by_model: dict[str, str | None]
│        │  含义: 各模型在该位置上的表达式
│        │  └─ <model_name>: str | None
│        └─ comparison_by_model: dict[str, ComparisonStats]
│           含义: 基准模型与其他模型逐一比较后的统计
│           └─ <compare_model_name>: dict
│              ├─ value_consist: bool
│              │  含义: 与基准模型的值是否一致
│              ├─ same_index: bool
│              │  含义: 行索引是否完全一致
│              ├─ nan_pattern_same: bool
│              │  含义: NaN 分布是否一致
│              ├─ left_only_rows: int
│              │  含义: 仅基准模型存在的行数
│              ├─ right_only_rows: int
│              │  含义: 仅对比模型存在的行数
│              ├─ overlap_rows: int
│              │  含义: 两者重叠的行数
│              ├─ max_abs_diff: float
│              │  含义: 重叠有效值中的最大绝对差
│              └─ tolerance: float
│                 含义: 本次比较使用的容忍阈值
└─ iteration_history: list[IterationSnapshot]
   含义: judge 级迭代历史
   └─ item: dict
      ├─ judge_iteration: int
      │  含义: 第几轮 judge
      ├─ decision: str
      │  含义: 这一轮的判决类型
      ├─ consistent: bool
      │  含义: 这一轮是否一致
      ├─ model_statuses: dict[str, str]
      │  含义: 这一轮每个模型的状态摘要
      │  └─ <model_name>: str
      ├─ final_factor_count: int
      │  含义: 这一轮已确认的最终因子数量
      ├─ issue_reason: str | None
      │  含义: 这一轮问题类型；一致时通常为 None
      └─ judge_feedback_used: str | None
         含义: 这一轮生成时用到的反馈文本



## Return of run.py

1. 运行级结果
   - decision.json
     保存完整的最终返回结果，方便追溯这次 Judge 的全部结论。
   - manifest.json
     保存轻量索引，后续程序优先读这个就行。
   - iteration_history.json
     保存 judge 迭代轨迹。
   - issue_report.json
     只有不一致时才会生成。
2. 因子级结果
   每个 confirmed factor 单独一个目录，里面保存：
   - metadata.json
     保存 factor_name、expression、core_logic、source_models 等元信息。
   - factor_values_long.parquet
     长表格式，适合和别的数据表 merge。
   - factor_values_wide.parquet
     宽表格式，适合直接做截面选股和组合构建。
   - backtest.parquet
     回测序列，方便后续分析和筛选。