import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    openclaw_host: str = os.getenv("OPENCLAW_HOST", "0.0.0.0")
    openclaw_port: int = int(os.getenv("OPENCLAW_PORT", "8765"))
    openclaw_db_path: str = os.getenv("OPENCLAW_DB_PATH", "~/.openclaw/kanban.db")
    openhands_url: str = os.getenv("OPENHANDS_URL", "http://localhost:8000")
    openhands_api_key: str = os.getenv("OPENHANDS_API_KEY", "")
    openhands_sandbox_dir: str = os.getenv("OPENHANDS_SANDBOX_DIR", "~/.openclaw/sandboxes/")
    nvidia_rpm_limit: int = int(os.getenv("NVIDIA_RPM_LIMIT", "40"))
    model_health_check_interval: int = int(os.getenv("MODEL_HEALTH_CHECK_INTERVAL", "60"))
    fallback_retry_max: int = int(os.getenv("FALLBACK_RETRY_MAX", "3"))
    nvidia_base_url: str = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
