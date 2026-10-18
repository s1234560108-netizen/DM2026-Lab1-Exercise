
"""Tests for tools_exploration.py using the official fixture."""

import json
import pytest
import numpy as np
import pandas as pd
import matplotlib.figure

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def _setup_dtm_session():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    loader = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    loader.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    dtm_tools = {
        t.name: t for t in make_dtm_tools(session)
    }
    dtm_tools["build_dtm_tool"].invoke({})

    exploration_tools = {
        t.name: t for t in make_exploration_tools(session)
    }
    return session, exploration_tools


def test_cosine_similarity_tool_known_answer_16():
    session, tools = _setup_dtm_session()
    cosine = tools["cosine_similarity_tool"]

    expected = [
        (0, 1, 0.9847),
        (0, 3, 0.0909),
        (3, 4, 0.9847),
        (0, 7, 0.0000),
    ]

    for i, j, expected_score in expected:
        result = cosine.invoke({
            "doc_idx_1": i,
            "doc_idx_2": j,
        })
        assert result["cosine_similarity"] == pytest.approx(
            expected_score, abs=1e-3
        )
        json.dumps(result)

    with pytest.raises(ValueError, match="out of bounds"):
        cosine.invoke({"doc_idx_1": -1, "doc_idx_2": 1})

    with pytest.raises(ValueError, match="out of bounds"):
        cosine.invoke({"doc_idx_1": 0, "doc_idx_2": 8})


def test_feature_correlation_matrix_tool_known_answer_19():
    session, tools = _setup_dtm_session()

    result = tools["feature_correlation_matrix_tool"].invoke({})

    expected_terms = [
        "alpha", "gamma", "delta", "beta", "always"
    ]

    expected_matrix = np.array([
        [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
        [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
        [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
        [0.2601, 0.3140, 0.3780, 0.2928, 1.0000],
    ])

    assert result["n_terms"] == 5
    assert result["term_names"] == expected_terms

    actual = np.asarray(result["correlation_matrix"])
    assert actual.shape == (5, 5)

    np.testing.assert_allclose(
        actual, expected_matrix, atol=1e-3
    )

    assert isinstance(
        session.pending_figure, matplotlib.figure.Figure
    )

    json.dumps(result)

    stored = session.results_store.get(result["result_id"])
    assert stored is not None
    assert isinstance(stored["full_report"], pd.DataFrame)


def test_exploration_tools_uninitialized():
    session = SessionState()
    tools = {
        t.name: t for t in make_exploration_tools(session)
    }

    with pytest.raises(ValueError, match="feature_matrix"):
        tools["cosine_similarity_tool"].invoke({
            "doc_idx_1": 0,
            "doc_idx_2": 1,
        })

    with pytest.raises(ValueError, match="feature_matrix"):
        tools["feature_correlation_matrix_tool"].invoke({})
