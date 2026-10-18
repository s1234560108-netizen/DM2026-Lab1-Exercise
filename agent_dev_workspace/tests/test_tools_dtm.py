"""Tests for tools_dtm.py using agent_dev/sample_fixture.csv."""

import json
import pytest
import numpy as np
import matplotlib.figure
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


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
    tools = make_tools(session)
    return session, {t.name: t for t in tools}


def test_build_dtm_tool_known_answer_4():
    """Test build_dtm_tool against Known Answer 4 in TEST_FIXTURE.md."""
    session, tools = _load_fixture_dataset()
    build_dtm_tool = tools["build_dtm_tool"]

    result = build_dtm_tool.invoke({})

    # Return summary assertions (Known Answer 4)
    assert result["n_documents"] == 8
    assert result["vocabulary_size"] == 5
    assert result["feature_names"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Session state assertions
    assert session.feature_matrix is not None
    assert issparse(session.feature_matrix)
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_matrix.nnz == 21
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]

    # Vectorizer artifact assertion
    vectorizer = session.artifacts.get("count_vectorizer")
    assert isinstance(vectorizer, CountVectorizer)
    assert list(vectorizer.get_feature_names_out()) == ["alpha", "always", "beta", "delta", "gamma"]


def test_build_dtm_tool_empty_vocabulary():
    """Test handling of valid min_df pruning that eliminates all vocabulary terms."""
    session, tools = _load_fixture_dataset()
    build_dtm_tool = tools["build_dtm_tool"]

    # min_df=8 requires a term in all 8 documents. Since row 7 is empty text,
    # no term appears in all 8 documents, triggering "After pruning, no terms remain".
    result = build_dtm_tool.invoke({"min_df": 8})

    assert result["n_documents"] == 8
    assert result["vocabulary_size"] == 0
    assert result["feature_names"] == []
    assert result["non_zero"] == 0
    assert result["total_elements"] == 0
    assert result["sparsity_pct"] == 0.0

    assert session.feature_matrix.shape == (8, 0)
    assert session.feature_names == []
    assert "count_vectorizer" not in session.artifacts


def test_build_dtm_tool_invalid_ngram_range():
    """Test validation of invalid ngram_range inputs."""
    session, tools = _load_fixture_dataset()
    build_dtm_tool = tools["build_dtm_tool"]

    # Invalid range: min > max
    with pytest.raises(ValueError, match="ngram_range must be a sequence of 2 positive integers"):
        build_dtm_tool.invoke({"ngram_range": [2, 1]})

    # Invalid format: not length 2
    with pytest.raises(ValueError, match="ngram_range must be a sequence of 2 positive integers"):
        build_dtm_tool.invoke({"ngram_range": [1, 2, 3]})


def test_term_frequency_tool_known_answer_14():
    """Test term_frequency_tool against Known Answer 14 in TEST_FIXTURE.md."""
    session, tools = _load_fixture_dataset()
    tools["build_dtm_tool"].invoke({})
    term_frequency_tool = tools["term_frequency_tool"]

    result = term_frequency_tool.invoke({})

    # Known Answer 14 expected term frequencies: alpha=6, always=7, beta=3, delta=4, gamma=7
    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }

    assert result["vocabulary_size"] == 5
    assert result["term_frequencies"] == expected_freqs

    # Verify JSON serializability of returned summary
    assert json.dumps(result) is not None

    # Verify full_report stored in session
    stored_entry = session.results_store.get(result["result_id"])
    assert stored_entry is not None
    assert "full_report" in stored_entry


def test_dtm_heatmap_tool_known_answer_21():
    """Test dtm_heatmap_tool against Known Answer 21 in TEST_FIXTURE.md."""
    session, tools = _load_fixture_dataset()
    tools["build_dtm_tool"].invoke({})
    dtm_heatmap_tool = tools["dtm_heatmap_tool"]

    # 1. Test default heatmap (returns all 8 documents and 5 terms)
    result_default = dtm_heatmap_tool.invoke({})

    assert result_default["sliced_documents"] == 8
    assert result_default["sliced_terms"] == 5
    assert result_default["terms_included"] == ["alpha", "always", "beta", "delta", "gamma"]

    # Verify figure set on session.pending_figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # Verify JSON serializability of default summary
    assert json.dumps(result_default) is not None

    # 2. Test corner slice: n_terms=2, n_documents=3 (Known Answer 21)
    result_corner = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert result_corner["sliced_documents"] == 3
    assert result_corner["sliced_terms"] == 2
    assert result_corner["terms_included"] == ["alpha", "always"]
    assert result_corner["matrix"] == [[3, 1], [2, 1], [1, 1]]

    # Verify JSON serializability of corner summary
    assert json.dumps(result_corner) is not None

    # Verify original DTM matrix remains unchanged
    assert session.feature_matrix.shape == (8, 5)


def test_dtm_tools_uninitialized_or_invalid():
    """Test calling DTM tools without prior build_dtm_tool or with invalid args."""
    session = SessionState()
    tools = {t.name: t for t in make_tools(session)}

    # Uninitialized DTM check
    with pytest.raises(ValueError, match="session.feature_matrix or session.feature_names is None"):
        tools["term_frequency_tool"].invoke({})

    with pytest.raises(ValueError, match="session.feature_matrix or session.feature_names is None"):
        tools["dtm_heatmap_tool"].invoke({})

    # Initialize DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    next(t for t in premade_tools if t.name == "load_dataset_tool").invoke(
        {"file_path": "agent_dev/sample_fixture.csv", "text_column": "text", "label_column": "label"}
    )
    tools["build_dtm_tool"].invoke({})

    # Invalid heatmap parameters (n_terms < 1 or n_documents < 1)
    with pytest.raises(ValueError, match="n_terms and n_documents must be positive integers"):
        tools["dtm_heatmap_tool"].invoke({"n_terms": 0, "n_documents": 5})

    with pytest.raises(ValueError, match="n_terms and n_documents must be positive integers"):
        tools["dtm_heatmap_tool"].invoke({"n_terms": 5, "n_documents": -1})
