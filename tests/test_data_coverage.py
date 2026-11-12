import pytest
from homefleet.data import assemble


def test_missing_historical_outage_truth_cannot_become_zero_outage_data(tmp_path, monkeypatch):
    monkeypatch.setattr(assemble, 'DATA_RAW', tmp_path)
    with pytest.raises(FileNotFoundError, match='refusing to create a zero-outage historical bundle'):
        assemble.build_metro_year('houston', 2024)
