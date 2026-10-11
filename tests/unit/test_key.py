import pytest
import pandas as pd
import behavioral_ml.features.key as key

def test_transform_key_events():
    sample = pd.read_parquet("tests/fixtures/transform_in.parquet")
    expected = pd.read_parquet("tests/fixtures/transform_out.parquet")

    result = key._transform_key_events(sample)

    result["Release Only"] = result["Release Only"].astype("boolean")
    expected["Release Only"] = expected["Release Only"].astype("boolean")

    pd.testing.assert_frame_equal(result, expected)