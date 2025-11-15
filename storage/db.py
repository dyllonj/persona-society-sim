"""Database adapter for structured logs (SQLite/Postgres)."""

from __future__ import annotations

from typing import Any, Iterable

try:  # pragma: no cover - optional dependency for persistence
    from sqlalchemy import text
    from sqlalchemy.engine import Engine, create_engine
except ModuleNotFoundError:  # pragma: no cover
    Engine = Any  # type: ignore

    def create_engine(_url: str):  # type: ignore
        raise ModuleNotFoundError("sqlalchemy is required for database logging")

    def text(statement: str) -> str:  # type: ignore
        return statement

DDL = """
CREATE TABLE IF NOT EXISTS agent_state (
  agent_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  persona_coeffs JSONB NOT NULL,
  steering_refs JSONB NOT NULL,
  active_alpha_overrides JSONB NOT NULL,
  system_prompt TEXT NOT NULL,
  location_id TEXT NOT NULL,
  status TEXT NOT NULL,
  goals JSONB NOT NULL,
  created_at TIMESTAMP NOT NULL,
  last_tick INT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_event (
  memory_id TEXT PRIMARY KEY,
  agent_id TEXT,
  kind TEXT NOT NULL,
  tick INT NOT NULL,
  ts TIMESTAMP NOT NULL,
  text TEXT NOT NULL,
  importance FLOAT,
  recency_decay FLOAT,
  embedding_id TEXT,
  source_msg_id TEXT
);
CREATE TABLE IF NOT EXISTS msg_log (
  msg_id TEXT PRIMARY KEY,
  run_id TEXT,
  tick INT,
  channel TEXT,
  from_agent TEXT,
  to_agent TEXT,
  room_id TEXT,
  content TEXT,
  tokens_in INT,
  tokens_out INT,
  temperature FLOAT,
  top_p FLOAT,
  steering_snapshot JSONB,
  layers_used INT[]
);
CREATE TABLE IF NOT EXISTS action_log (
  action_id TEXT PRIMARY KEY,
  run_id TEXT,
  tick INT,
  agent_id TEXT,
  action_type TEXT,
  params JSONB,
  outcome TEXT,
  info JSONB,
  prompt_text TEXT,
  prompt_hash TEXT,
  plan_metadata JSONB,
  reflection_summary TEXT,
  reflection_implications JSONB
);
CREATE TABLE IF NOT EXISTS econ_txn (
  txn_id TEXT PRIMARY KEY,
  run_id TEXT,
  tick INT,
  buyer_id TEXT,
  seller_id TEXT,
  item TEXT,
  qty INT,
  price FLOAT
);
CREATE TABLE IF NOT EXISTS vote_log (
  vote_id TEXT PRIMARY KEY,
  run_id TEXT,
  tick INT,
  proposal_id TEXT,
  agent_id TEXT,
  vote TEXT
);
CREATE TABLE IF NOT EXISTS sanction_log (
  sanction_id TEXT PRIMARY KEY,
  run_id TEXT,
  tick INT,
  actor_id TEXT,
  target_id TEXT,
  kind TEXT,
  justification TEXT,
  amount FLOAT
);
CREATE TABLE IF NOT EXISTS safety_event (
  event_id TEXT PRIMARY KEY,
  run_id TEXT,
  tick INT,
  agent_id TEXT,
  kind TEXT,
  severity TEXT,
  applied_alpha_delta JSONB
);
CREATE TABLE IF NOT EXISTS graph_snapshot (
  run_id TEXT,
  tick INT,
  edges JSONB,
  centrality JSONB,
  PRIMARY KEY (run_id, tick)
);
CREATE TABLE IF NOT EXISTS steering_vector_store (
  vector_store_id TEXT,
  trait TEXT,
  method TEXT,
  layer_id INT,
  vector_path TEXT,
  pos_set_hash TEXT,
  neg_set_hash TEXT,
  created_at TIMESTAMP,
  PRIMARY KEY (vector_store_id, layer_id)
);
CREATE TABLE IF NOT EXISTS run_config (
  run_id TEXT PRIMARY KEY,
  git_commit TEXT,
  model_name TEXT,
  layers INT[],
  population INT,
  steps INT,
  scenario TEXT,
  seed INT,
  steering JSONB,
  notes TEXT
);
CREATE TABLE IF NOT EXISTS run_summary (
  run_id TEXT PRIMARY KEY,
  started_at TIMESTAMP,
  finished_at TIMESTAMP,
  tokens_in BIGINT,
  tokens_out BIGINT,
  crashes INT
);
"""


class Database:
    def __init__(self, url: str):
        self.url = url
        self.engine: Engine = create_engine(url)

    def init(self) -> None:
        with self.engine.begin() as conn:
            for statement in filter(None, DDL.split(";")):
                conn.execute(text(statement))

    def insert_many(self, table: str, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        columns = rows[0].keys()
        cols_sql = ", ".join(columns)
        vals_sql = ", ".join(f":{col}" for col in columns)
        statement = text(f"INSERT INTO {table} ({cols_sql}) VALUES ({vals_sql})")
        with self.engine.begin() as conn:
            conn.execute(statement, rows)
