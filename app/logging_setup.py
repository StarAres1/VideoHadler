"""Загрузка конфигурации логирования из JSON (см. config/logging.json)."""

from __future__ import annotations

import json
import logging
import logging.config
import os
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_handler_paths(config: dict, base: Path) -> None:
    for handler in config.get("handlers", {}).values():
        if not isinstance(handler, dict):
            continue
        fn = handler.get("filename")
        if fn:
            path = Path(fn)
            if not path.is_absolute():
                handler["filename"] = str(base / path)


def _ensure_log_directories(config: dict) -> None:
    for handler in config.get("handlers", {}).values():
        if not isinstance(handler, dict):
            continue
        fn = handler.get("filename")
        if fn:
            Path(fn).parent.mkdir(parents=True, exist_ok=True)


def _apply_env_level_override(config: dict) -> None:
    """Переопределяет уровень логгера app через переменную APP_LOG_LEVEL (DEBUG, INFO, WARNING, ERROR)."""
    name = os.environ.get("APP_LOG_LEVEL", "").strip().upper()
    if not name or not hasattr(logging, name):
        return
    loggers = config.setdefault("loggers", {})
    entry = loggers.setdefault("app", {})
    entry["level"] = name


def setup_application_logging(config_path: str | os.PathLike[str] | None = None) -> Path | None:
    """
    Инициализирует логирование для пакета app (логгер app и иерархия app.*).

    Параметры окружения:
      APP_LOG_CONFIG — путь к JSON с dictConfig (по умолчанию config/logging.json в корне проекта).
      APP_LOG_LEVEL   — уровень для логгера app (например DEBUG), без правки файла.

    Возвращает путь к использованному файлу конфигурации или None при fallback.
    """
    root = _project_root()
    default_cfg = root / "config" / "logging.json"
    path = Path(os.environ.get("APP_LOG_CONFIG", config_path or default_cfg))
    if not path.is_file():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
        logging.getLogger("app").warning("Файл конфигурации логирования не найден: %s", path)
        return None

    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("app").error("Не удалось прочитать %s: %s", path, e)
        return None

    _resolve_handler_paths(config, root)
    _apply_env_level_override(config)
    _ensure_log_directories(config)

    logging.config.dictConfig(config)
    return path
