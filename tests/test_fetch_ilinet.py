from unittest.mock import Mock, patch

import pandas as pd
import pytest

from epicast.ingest.fetch_ilinet import EpidataError, clean_weekly_series, fetch_ilinet


def test_clean_weekly_series_dedupes_revisions_and_sets_calendar_date():
    raw = pd.DataFrame(
        [
            {"epiweek": 201501, "issue": 201501, "wili": 1.1, "ili": 1.0, "num_ili": 100, "num_patients": 10000},
            {"epiweek": 201501, "issue": 201510, "wili": 1.3, "ili": 1.2, "num_ili": 120, "num_patients": 10000},
            {"epiweek": 201502, "issue": 201502, "wili": 1.4, "ili": 1.3, "num_ili": 130, "num_patients": 10000},
        ]
    )

    clean = clean_weekly_series(raw)

    assert len(clean) == 2
    assert list(clean.columns) == ["epiweek", "week_start", "wili", "ili", "num_ili", "num_patients"]
    # epiweek 201501's later issue (revision) should win.
    assert clean.loc[clean["epiweek"] == 201501, "wili"].iloc[0] == 1.3
    assert clean["week_start"].is_monotonic_increasing


def test_fetch_ilinet_raises_on_non_success_result():
    mock_response = Mock()
    mock_response.json.return_value = {"result": -2, "message": "no results"}
    mock_response.raise_for_status = Mock()

    with patch("epicast.ingest.fetch_ilinet.requests.get", return_value=mock_response):
        with pytest.raises(EpidataError):
            fetch_ilinet()


def test_fetch_ilinet_returns_dataframe_of_epidata_records():
    mock_response = Mock()
    mock_response.json.return_value = {
        "result": 1,
        "epidata": [{"epiweek": 201501, "issue": 201501, "wili": 1.1}],
    }
    mock_response.raise_for_status = Mock()

    with patch("epicast.ingest.fetch_ilinet.requests.get", return_value=mock_response) as mock_get:
        df = fetch_ilinet(region="nat", epiweeks="201501-201501")

    assert len(df) == 1
    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs["params"] == {"regions": "nat", "epiweeks": "201501-201501"}
