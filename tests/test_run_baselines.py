from scripts.run_baselines import _parquet_dir_for_run, _sqlite_db_url_for_run


def test_baseline_sqlite_url_isolated_per_run():
    assert (
        _sqlite_db_url_for_run("sqlite:///./storage/sim_debug32.db", "debug32-targeted")
        == "sqlite:///storage/sim_debug32-debug32-targeted.db"
    )


def test_baseline_parquet_dir_uses_sibling_run_dir():
    assert (
        _parquet_dir_for_run("./storage/dumps/debug32", "debug32", "debug32-targeted")
        == "storage/dumps/debug32-targeted"
    )


def test_baseline_non_sqlite_url_is_preserved():
    url = "postgresql+psycopg://persona:persona@localhost:5432/persona"
    assert _sqlite_db_url_for_run(url, "debug32-targeted") == url
