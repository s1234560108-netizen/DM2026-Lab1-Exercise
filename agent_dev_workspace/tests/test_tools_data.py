"""Tests for tools_data.py using agent_dev/sample_fixture.csv."""

import pytest
import pandas as pd
import numpy as np
import matplotlib.figure

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools
from tools_dtm import make_tools as make_dtm_tools


def _load_fixture_dataset():
    """Helper to initialize SessionState and load sample_fixture.csv."""
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
    tools = make_data_tools(session)
    return session, {t.name: t for t in tools}


def test_check_missing_tool_known_answer_2():
    """Test check_missing_tool with drop=False against Known Answer 2 in TEST_FIXTURE.md."""
    session, tools = _load_fixture_dataset()
    check_missing_tool = tools["check_missing_tool"]

    result = check_missing_tool.invoke({"drop": False})

    # Known Answer 2: exactly 1 row has missing/empty text (index 7)
    assert result["total_rows"] == 8
    assert result["missing_count"] == 1
    assert result["missing_indices"] == [7]
    assert result["remaining_rows"] == 8
    assert len(session.dataframe) == 8


def test_check_missing_tool_drop():
    """Test check_missing_tool with drop=True, state cleanup, and label resync."""
    session, tools = _load_fixture_dataset()
    check_missing_tool = tools["check_missing_tool"]

    # Build DTM first to ensure state invalidation can be tested
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    assert session.feature_matrix is not None
    assert "count_vectorizer" in session.artifacts

    result = check_missing_tool.invoke({"drop": True})

    assert result["missing_count"] == 1
    assert result["remaining_rows"] == 7
    assert len(session.dataframe) == 7

    # Verify stale DTM state and vectorizer artifact are invalidated
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert "count_vectorizer" not in session.artifacts

    # Verify session labels and categories resynchronization
    assert len(session.labels) == 7
    assert sorted(list(session.categories)) == ["catA", "catB"]


def test_check_duplicates_tool_known_answer_3():
    """Test check_duplicates_tool with drop=False against Known Answer 3 in TEST_FIXTURE.md."""
    session, tools = _load_fixture_dataset()
    check_duplicates_tool = tools["check_duplicates_tool"]

    result = check_duplicates_tool.invoke({"drop": False})

    # Known Answer 3: exactly 1 duplicate occurrence (row 6 duplicate of row 5)
    assert result["total_rows"] == 8
    assert result["duplicate_occurrences"] == 1
    assert result["duplicated_indices"] == [6]
    assert result["remaining_rows"] == 8
    assert len(session.dataframe) == 8


def test_check_duplicates_tool_drop_keep_false():
    """Test check_duplicates_tool with drop=True (keep=False) leaving 6 rows per Known Answer 3."""
    session, tools = _load_fixture_dataset()
    check_duplicates_tool = tools["check_duplicates_tool"]

    # Build DTM first to verify state invalidation on drop
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    result = check_duplicates_tool.invoke({"drop": True})

    # Known Answer 3: keep=False drops BOTH copies of duplicates, leaving exactly 6 rows
    assert result["duplicate_occurrences"] == 1
    assert result["remaining_rows"] == 6
    assert len(session.dataframe) == 6

    # Verify stale DTM state invalidation and label resync
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert "count_vectorizer" not in session.artifacts
    assert len(session.labels) == 6


def test_data_tools_uninitialized_session():
    """Test calling data tools on empty or uninitialized session."""
    session = SessionState()
    tools = {t.name: t for t in make_data_tools(session)}

    with pytest.raises(ValueError, match="session.dataframe is empty or None"):
        tools["check_missing_tool"].invoke({})

    with pytest.raises(ValueError, match="session.dataframe is empty or None"):
        tools["check_duplicates_tool"].invoke({})


def test_list_files_tool_known_answer_11():
    """Test list_files_tool against Known Answer 11 in TEST_FIXTURE.md."""
    session = SessionState()
    tools = {t.name: t for t in make_data_tools(session)}
    list_files_tool = tools["list_files_tool"]

    # Known Answer 11: subdirectory="newdataset" returns files=["Reddit-stock-sentiment.csv"]
    result = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert result["directories"] == []
    assert "Reddit-stock-sentiment.csv" in result["files"]

    # Test directory traversal prevention
    with pytest.raises(ValueError, match="Access denied"):
        list_files_tool.invoke({"subdirectory": "../.."})


def test_inspect_data_tool():
    """Test inspect_data_tool on sample fixture dataset."""
    session, tools = _load_fixture_dataset()
    inspect_data_tool = tools["inspect_data_tool"]

    # Add a long string to test preview truncation
    session.dataframe.loc[0, "long_text"] = "A" * 300

    result = inspect_data_tool.invoke({"n_rows": 3})

    assert result["total_rows"] == 8
    assert "text" in result["columns"]
    assert result["preview_rows"] == 3
    assert len(result["preview"]) == 3

    # Check that long_text is truncated in preview
    assert result["preview"][0]["long_text"] == ("A" * 200) + "..."
    # Confirm underlying session.dataframe is untouched
    assert session.dataframe.loc[0, "long_text"] == "A" * 300


def test_tokenize_tool_known_answer_10():
    """Test tokenize_tool against Known Answer 10 in TEST_FIXTURE.md."""
    session, tools = _load_fixture_dataset()
    tokenize_tool = tools["tokenize_tool"]

    result = tokenize_tool.invoke({})

    # Known Answer 10 expected token lists
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],  # Empty document 7
    ]

    assert result["total_documents"] == 8
    assert "token_preview" in result

    # Verify complete tokenized documents stored in dataframe and session.artifacts
    assert "unigrams" in session.dataframe.columns
    assert session.dataframe["unigrams"].tolist() == expected_tokens
    assert session.artifacts["unigrams"] == expected_tokens


def test_sample_data_tool_known_answer_13():
    """Test sample_data_tool against Known Answer 13 in TEST_FIXTURE.md."""
    session, tools = _load_fixture_dataset()
    sample_data_tool = tools["sample_data_tool"]

    # Known Answer 13: n=4, random_state=42 returns sampled_indices [1, 5, 0, 7] in exact order
    result = sample_data_tool.invoke({"n": 4, "random_state": 42, "replace_dataframe": False})

    assert result["sampled_rows"] == 4
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 8  # replace_dataframe=False preserves original DataFrame

    # Build DTM first to test state cleanup on replace_dataframe=True
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    assert session.feature_matrix is not None

    result_replace = sample_data_tool.invoke({"n": 4, "random_state": 42, "replace_dataframe": True})

    assert result_replace["sampled_rows"] == 4
    assert len(session.dataframe) == 4

    # Verify stale DTM state invalidation and label resynchronization
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert len(session.labels) == 4


def test_describe_data_tool_known_answer_18():
    """Test describe_data_tool against Known Answer 18 in TEST_FIXTURE.md."""
    session, tools = _load_fixture_dataset()
    describe_data_tool = tools["describe_data_tool"]

    result = describe_data_tool.invoke({"text_column": "text", "category_column": "category_name"})

    # Known Answer 18 document text lengths: [29, 23, 17, 30, 24, 18, 18, 0]
    expected_lengths = [29, 23, 17, 30, 24, 18, 18, 0]
    assert "text_length" in session.dataframe.columns
    assert session.dataframe["text_length"].tolist() == expected_lengths

    # Verify overall .describe() stats
    overall = result["overall"]
    assert overall["count"] == pytest.approx(8, abs=1e-3)
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == pytest.approx(0.0, abs=1e-3)
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == pytest.approx(30.0, abs=1e-3)

    # Verify catA .describe() stats
    catA = result["by_category"]["catA"]
    assert catA["count"] == pytest.approx(4, abs=1e-3)
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.50, abs=1e-3)
    assert catA["min"] == pytest.approx(0.0, abs=1e-3)
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.00, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.50, abs=1e-3)
    assert catA["max"] == pytest.approx(29.0, abs=1e-3)

    # Verify catB .describe() stats
    catB = result["by_category"]["catB"]
    assert catB["count"] == pytest.approx(4, abs=1e-3)
    assert catB["mean"] == pytest.approx(22.50, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == pytest.approx(18.0, abs=1e-3)
    assert catB["25%"] == pytest.approx(18.00, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.00, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.50, abs=1e-3)
    assert catB["max"] == pytest.approx(30.0, abs=1e-3)

    # Verify Matplotlib figure set on session.pending_figure and x-axis category tick labels
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
    ax = session.pending_figure.axes[0]
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    assert set(xticklabels) == {"catA", "catB"}

    # Verify JSON serializability of returned dictionary
    import json
    json_str = json.dumps(result)
    assert isinstance(json_str, str)


def test_describe_data_tool_uninitialized():
    """Test describe_data_tool on an uninitialized session."""
    session = SessionState()
    tools = {t.name: t for t in make_data_tools(session)}

    with pytest.raises(ValueError, match="session.dataframe is empty or None"):
        tools["describe_data_tool"].invoke({})


def test_describe_data_tool_missing_column_and_missing_category():
    """Test describe_data_tool error handling for missing text column and fallback when category_column is absent."""
    session, tools = _load_fixture_dataset()
    describe_data_tool = tools["describe_data_tool"]

    # Test error when text_column does not exist
    with pytest.raises(ValueError, match="Column 'non_existent' not found"):
        describe_data_tool.invoke({"text_column": "non_existent"})

    # Test fallback behavior when category_column does not exist in dataframe
    result = describe_data_tool.invoke({"text_column": "text", "category_column": "non_existent_category"})
    assert result["by_category"] == {}
    assert result["overall"]["count"] == 8
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
