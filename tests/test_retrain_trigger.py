from unittest.mock import MagicMock

from epicast.monitor import retrain_trigger


def _run_main(monkeypatch, argv, drifted: bool):
    monkeypatch.setattr(retrain_trigger, "ingest", MagicMock())
    monkeypatch.setattr(retrain_trigger, "generate_report", MagicMock(return_value=(MagicMock(), drifted)))
    mock_retrain = MagicMock()
    monkeypatch.setattr(retrain_trigger, "retrain", mock_retrain)
    monkeypatch.setattr("sys.argv", ["retrain_trigger", *argv])

    retrain_trigger.main()
    return mock_retrain


def test_retrains_when_drift_detected(monkeypatch):
    mock_retrain = _run_main(monkeypatch, ["--skip-ingest"], drifted=True)

    mock_retrain.assert_called_once()


def test_skips_retrain_when_no_drift(monkeypatch):
    mock_retrain = _run_main(monkeypatch, ["--skip-ingest"], drifted=False)

    mock_retrain.assert_not_called()


def test_force_retrains_even_without_drift(monkeypatch):
    mock_retrain = _run_main(monkeypatch, ["--skip-ingest", "--force"], drifted=False)

    mock_retrain.assert_called_once()


def test_skip_ingest_flag_skips_ingestion(monkeypatch):
    mock_ingest = MagicMock()
    monkeypatch.setattr(retrain_trigger, "ingest", mock_ingest)
    monkeypatch.setattr(retrain_trigger, "generate_report", MagicMock(return_value=(MagicMock(), False)))
    monkeypatch.setattr(retrain_trigger, "retrain", MagicMock())
    monkeypatch.setattr("sys.argv", ["retrain_trigger", "--skip-ingest"])

    retrain_trigger.main()

    mock_ingest.assert_not_called()
