"""Configuration management. Secrets come only from environment / .env.
Never hardcode keys, never log or render them."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        # Minimal fallback parser (no secrets printed)
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

@dataclass
class Settings:
    llm_provider: str = "none"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"
    news_api_key: str = ""
    risk_free_rate: float = 0.06
    trading_days_per_year: int = 252
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    request_timeout_seconds: int = 30
    max_retries: int = 3

    @property
    def raw_dir(self) -> Path: return self.data_dir / "raw"
    @property
    def processed_dir(self) -> Path: return self.data_dir / "processed"
    @property
    def metadata_dir(self) -> Path: return self.data_dir / "metadata"

    @property
    def has_llm(self) -> bool:
        if self.llm_provider == "gemini": return bool(self.gemini_api_key)
        if self.llm_provider == "openai": return bool(self.openai_api_key)
        return False

    def validate(self) -> list[str]:
        issues = []
        if self.llm_provider not in ("gemini", "openai", "none"):
            issues.append(f"Unknown LLM_PROVIDER '{self.llm_provider}' (use gemini, openai or none).")
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            issues.append("GEMINI_API_KEY not set — AI assistant will run in rule-based fallback mode; all local analytics still work.")
        if self.llm_provider == "openai" and not self.openai_api_key:
            issues.append("OPENAI_API_KEY not set — AI assistant will run in rule-based fallback mode; all local analytics still work.")
        if not (0.0 <= self.risk_free_rate <= 0.30):
            issues.append("RISK_FREE_RATE looks out of range (expected a decimal like 0.06).")
        return issues

    def redacted_summary(self) -> dict:
        """Safe-to-display summary — keys are never included."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model if self.has_llm else "(fallback: rule-based)",
            "llm_configured": self.has_llm,
            "risk_free_rate": self.risk_free_rate,
            "trading_days_per_year": self.trading_days_per_year,
            "data_dir": str(self.data_dir),
        }

def get_settings() -> Settings:
    _load_dotenv()
    data_dir = Path(os.getenv("FINSIGHT_DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "none").strip().lower(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.0-flash").strip(),
        news_api_key=os.getenv("NEWS_API_KEY", "").strip(),
        risk_free_rate=float(os.getenv("RISK_FREE_RATE", "0.06")),
        trading_days_per_year=int(os.getenv("TRADING_DAYS_PER_YEAR", "252")),
        data_dir=data_dir,
    )
