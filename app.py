# callcenter-intelligence
# app.py

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.agents.transcription import _get_whisper_model
from src.database.connection import get_engine, init_db
from src.graph.workflow import compile_workflow
from src.security.audit import AuditLogger
from src.ui.app import build_app
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("callcenter")


def main() -> None:
    config = load_config()
    engine = get_engine(config.db_path, config.db_encryption_key)
    init_db(engine)
    logger.info("loading whisper model size=%s", config.whisper_model_size)
    _get_whisper_model(config.whisper_model_size)
    workflow = compile_workflow(config, engine, AuditLogger(engine))
    demo = build_app(workflow, engine, config.confidence_threshold)
    # Bind all interfaces inside any container, not just Spaces: a published port
    # cannot reach a process listening on loopback, so `docker run -p 7860:7860`
    # would appear to start and then refuse every connection.
    in_container = bool(os.getenv("SPACE_ID")) or Path("/.dockerenv").exists()
    host = os.getenv("SERVER_HOST") or ("0.0.0.0" if in_container else "127.0.0.1")
    demo.launch(server_name=host, server_port=7860)


if __name__ == "__main__":
    main()
