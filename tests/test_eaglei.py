import pytest
from homefleet.paths import DATA_RAW
from homefleet.data.eaglei import load_tx, load_mcc, county_fraction_panel

pytestmark = pytest.mark.skipif(not (DATA_RAW / "eaglei" / "eaglei_2023_TX.csv").exists(), reason="raw eaglei not present")


def test_2023_header_quirk():
    df = load_tx(2023)
    assert "customers_out" in df.columns and df.t.dt.tz is not None


def test_fraction_panel_bounds():
    frac = county_fraction_panel([2024], fips=["48201"])
    assert frac["48201"].between(0, 1).all()
    # beryl: harris county mass outage july 8 2024
    assert frac.loc["2024-07-08", "48201"].max() > 0.5
