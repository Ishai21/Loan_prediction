"""Tests for the two-stage inference layer."""

import pandas as pd
import pytest

from loan_model import LoanApprovalApp

APPLICANT = {
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 9600000.0,
    "loan_amount": 29900000.0,
    "loan_term": 12,
    "cibil_score": 778,
    "residential_assets_value": 2400000.0,
    "commercial_assets_value": 17600000.0,
    "luxury_assets_value": 22700000.0,
    "bank_asset_value": 8000000.0,
}


@pytest.fixture(scope="module")
def model():
    return LoanApprovalApp()


def test_both_pipelines_load(model):
    assert model.clf is not None
    assert model.reg is not None


def test_predict_returns_expected_shape(model):
    result = model.two_stage_predict(pd.DataFrame([APPLICANT]))

    assert set(result) == {"approve", "regression_prediction"}
    assert result["approve"] in (0, 1)


def test_stage_two_runs_only_when_approved(model):
    result = model.two_stage_predict(pd.DataFrame([APPLICANT]))

    if result["approve"] == 1:
        assert isinstance(result["regression_prediction"], float)
        assert result["regression_prediction"] > 0
    else:
        assert result["regression_prediction"] is None


def test_low_cibil_is_rejected(model):
    """CIBIL score is the dominant predictor -- a very low score should reject."""
    applicant = {**APPLICANT, "cibil_score": 300}

    result = model.two_stage_predict(pd.DataFrame([applicant]))

    assert result["approve"] == 0
    assert result["regression_prediction"] is None
