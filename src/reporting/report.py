"""Report generation from *computed* results only."""
from __future__ import annotations
import io
import pandas as pd

def build_markdown_report(title: str, summary: dict, sections: dict[str, dict],
                          quality: dict | None = None) -> str:
    lines = [f"# {title}", "", "## Executive summary"]
    for k, v in summary.items():
        lines.append(f"- **{k}:** {v}")
    for name, metrics in sections.items():
        lines += ["", f"## {name}", ""]
        for k, v in metrics.items():
            if isinstance(v, (dict, list)):
                continue
            lines.append(f"- **{k}:** {v}")
    lines += ["", "## Methodology",
              "- Simple close-to-close returns on unadjusted bhavcopy prices unless stated otherwise.",
              "- Annualization uses 252 trading days. Sharpe uses the configured risk-free rate.",
              "- Portfolio returns assume daily rebalancing to target weights unless stated otherwise.",
              "- Technical indicators and ML outputs are descriptive/experimental, not predictions.",
              "", "## Limitations",
              "- Historical prices are unadjusted: splits/bonuses can create artificial jumps (flagged in data quality).",
              "- Historical risk and optimized allocations are sample-period estimates, not guarantees.",
              "- This report is informational research, not investment advice. No trades are executed."]
    if quality:
        lines += ["", "## Data-quality notes"]
        for k, v in quality.items():
            lines.append(f"- **{k}:** {v}")
    return "\n".join(lines)

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
