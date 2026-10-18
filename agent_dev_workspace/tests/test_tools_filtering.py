"""Tests for tools_filtering.py using agent_dev/sample_fixture.csv."""

import pytest
import numpy as np
import pandas as pd

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def _setup_dtm_session():
    """Helper to initialize SessionState, load sample_fixture.csv, and build DTM."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    return session, variance_filter_tool, pearson_filter_tool, spearman_filter_tool


def test_variance_filter_tool_known_answer_5():
    """Test variance_filter_tool against Known Answer 5 in TEST_FIXTURE.md."""
    session, variance_filter_tool, _, _ = _setup_dtm_session()

    # Save original feature matrix shape and data for immutability check
    orig_matrix_shape = session.feature_matrix.shape
    orig_matrix_data = session.feature_matrix.toarray().copy()
    orig_feature_names = list(session.feature_names)

    # Execute variance_filter_tool with threshold=0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Assert summary filtering results
    assert result["n_terms_before"] == 5
    assert result["n_terms_kept"] == 4
    assert result["n_terms_removed"] == 1
    assert set(result["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert result["removed_terms"] == ["always"]

    # Retrieve full_report from stored result
    stored_entry = session.results_store.get(result["result_id"])
    assert stored_entry is not None
    full_report = stored_entry["full_report"]
    assert isinstance(full_report, pd.DataFrame)

    # Check full_report columns and order
    assert list(full_report.columns) == ["term", "variance"]

    # Check sorting: variance should be in descending order
    assert full_report["variance"].is_monotonic_decreasing

    # Map term variances from full_report
    report_variances = dict(zip(full_report["term"], full_report["variance"]))

    # Verify exact Known Answer 5 term variances with numerical tolerance
    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375,
    }

    for term, exp_val in expected_variances.items():
        assert term in report_variances
        assert report_variances[term] == pytest.approx(exp_val, abs=1e-3)

    # Immutability assertion: feature matrix and names must remain unchanged
    assert session.feature_matrix.shape == orig_matrix_shape
    assert np.array_equal(session.feature_matrix.toarray(), orig_matrix_data)
    assert session.feature_names == orig_feature_names


def test_variance_filter_tool_invalid_threshold():
    """Test validation of negative threshold."""
    session, variance_filter_tool, _, _ = _setup_dtm_session()

    with pytest.raises(ValueError, match="threshold must be a non-negative float"):
        variance_filter_tool.invoke({"threshold": -0.1})


def test_variance_filter_tool_uninitialized_dtm():
    """Test calling variance_filter_tool before running build_dtm_tool."""
    session = SessionState()
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    with pytest.raises(ValueError, match="session.feature_matrix or session.feature_names is None"):
        variance_filter_tool.invoke({"threshold": 0.15})


def test_pearson_filter_tool_known_answer_6():
    """Test pearson_filter_tool against Known Answer 6 in TEST_FIXTURE.md."""
    session, _, pearson_filter_tool, _ = _setup_dtm_session()

    # Save original matrix & names for immutability assertion
    orig_matrix_shape = session.feature_matrix.shape
    orig_matrix_data = session.feature_matrix.toarray().copy()
    orig_feature_names = list(session.feature_names)

    # Run pearson_filter_tool with target_class="catB" and threshold=0.0
    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert result["target_class"] == "catB"
    assert result["n_terms_before"] == 5
    assert result["n_terms_kept"] == 5
    assert result["n_terms_removed"] == 0

    # Retrieve stored full_report
    stored_entry = session.results_store.get(result["result_id"])
    assert stored_entry is not None
    full_report = stored_entry["full_report"]
    assert isinstance(full_report, pd.DataFrame)

    # Check full_report columns and descending sorting by pearson_r
    assert list(full_report.columns[:2]) == ["term", "pearson_r"]
    assert full_report["pearson_r"].is_monotonic_decreasing

    report_rs = dict(zip(full_report["term"], full_report["pearson_r"]))

    # Expected Pearson correlation values per Known Answer 6
    expected_pearson = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    for term, exp_r in expected_pearson.items():
        assert term in report_rs
        assert report_rs[term] == pytest.approx(exp_r, abs=1e-3)

    # Test thresholding at |r| > 0.8
    result_high = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.8})
    assert set(result_high["kept_terms"]) == {"delta", "gamma"}
    assert set(result_high["removed_terms"]) == {"alpha", "always", "beta"}

    # DTM Immutability assertion
    assert session.feature_matrix.shape == orig_matrix_shape
    assert np.array_equal(session.feature_matrix.toarray(), orig_matrix_data)
    assert session.feature_names == orig_feature_names


def test_pearson_filter_tool_invalid_inputs():
    """Test invalid target_class and invalid threshold for pearson_filter_tool."""
    session, _, pearson_filter_tool, _ = _setup_dtm_session()

    # Invalid target_class
    with pytest.raises(ValueError, match="target_class 'nonexistent' not found"):
        pearson_filter_tool.invoke({"target_class": "nonexistent", "threshold": 0.0})

    # Negative threshold
    with pytest.raises(ValueError, match="threshold must be a non-negative float"):
        pearson_filter_tool.invoke({"target_class": "catB", "threshold": -0.5})


def test_spearman_filter_tool_known_answer_6():
    """Test spearman_filter_tool against Known Answer 6 in TEST_FIXTURE.md."""
    session, _, _, spearman_filter_tool = _setup_dtm_session()

    # Save original matrix & names for immutability assertion
    orig_matrix_shape = session.feature_matrix.shape
    orig_matrix_data = session.feature_matrix.toarray().copy()
    orig_feature_names = list(session.feature_names)

    # Run spearman_filter_tool with target_class="catB" and threshold=0.0
    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert result["target_class"] == "catB"
    assert result["n_terms_before"] == 5
    assert result["n_terms_kept"] == 5
    assert result["n_terms_removed"] == 0

    # Retrieve stored full_report
    stored_entry = session.results_store.get(result["result_id"])
    assert stored_entry is not None
    full_report = stored_entry["full_report"]
    assert isinstance(full_report, pd.DataFrame)

    # Check full_report columns and descending sorting by spearman_r
    assert list(full_report.columns[:2]) == ["term", "spearman_r"]
    assert full_report["spearman_r"].is_monotonic_decreasing

    report_rhos = dict(zip(full_report["term"], full_report["spearman_r"]))

    # Expected Spearman correlation values per Known Answer 6
    expected_spearman = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    for term, exp_rho in expected_spearman.items():
        assert term in report_rhos
        assert report_rhos[term] == pytest.approx(exp_rho, abs=1e-3)

    # Test thresholding at |rho| > 0.8
    result_high = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.8})
    assert set(result_high["kept_terms"]) == {"delta", "gamma"}
    assert set(result_high["removed_terms"]) == {"alpha", "always", "beta"}

    # DTM Immutability assertion
    assert session.feature_matrix.shape == orig_matrix_shape
    assert np.array_equal(session.feature_matrix.toarray(), orig_matrix_data)
    assert session.feature_names == orig_feature_names


def test_spearman_filter_tool_invalid_inputs():
    """Test invalid target_class and invalid threshold for spearman_filter_tool."""
    session, _, _, spearman_filter_tool = _setup_dtm_session()

    # Invalid target_class
    with pytest.raises(ValueError, match="target_class 'nonexistent' not found"):
        spearman_filter_tool.invoke({"target_class": "nonexistent", "threshold": 0.0})

    # Negative threshold
    with pytest.raises(ValueError, match="threshold must be a non-negative float"):
        spearman_filter_tool.invoke({"target_class": "catB", "threshold": -0.5})
