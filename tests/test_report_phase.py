import os
import json
import tempfile
import types
import config
from utils.tools import register_team, start_report_phase, generate_html_report

class DummyTeam:
    def __init__(self):
        self.report_phase = False
    def start_report_phase(self):
        self.report_phase = True

def setup_env(monkeypatch):
    tmpdir = tempfile.TemporaryDirectory()
    gen_dir = os.path.join(tmpdir.name, "Generated_Files")
    os.makedirs(gen_dir, exist_ok=True)
    monkeypatch.setattr(config, "GENERATED_FILES_DIR", gen_dir, raising=False)
    monkeypatch.chdir(tmpdir.name)
    team = DummyTeam()
    register_team(team)
    return tmpdir, gen_dir, team


def create_requirements(gen_dir):
    for name in [
        "stock_data.csv",
        "processed_data.csv",
        "trained_model.pkl",
        "evaluation.json",
        "analysis_chart.png",
        "quality_report.json",
    ]:
        with open(os.path.join(gen_dir, name), "w") as f:
            if name.endswith(".json"):
                json.dump({}, f)
            else:
                f.write("data")


def create_tasks(gen_dir, status="completed"):
    tasks = {
        "t1": {"status": status},
        "t2": {"status": status},
    }
    with open(os.path.join(gen_dir, "tasks.json"), "w") as f:
        json.dump(tasks, f)


def test_start_report_phase_checks(monkeypatch):
    tmpdir, gen_dir, team = setup_env(monkeypatch)
    try:
        create_requirements(gen_dir)
        create_tasks(gen_dir, status="in_progress")
        result = start_report_phase()
        assert "error" in result
        create_tasks(gen_dir, status="completed")
        result = start_report_phase()
        assert result == {"success": True}
        assert team.report_phase
    finally:
        tmpdir.cleanup()


def test_generate_html_requires_phase(monkeypatch):
    tmpdir, gen_dir, team = setup_env(monkeypatch)
    try:
        result = generate_html_report()
        assert "error" in result
        create_requirements(gen_dir)
        create_tasks(gen_dir)
        start_report_phase()
        result = generate_html_report()
        assert result.get("success")
        assert os.path.exists(os.path.join(gen_dir, "investor_report.html"))
    finally:
        tmpdir.cleanup()
