"""Tests for mine_patterns_tool in tools_patterns.py using agent_dev/sample_fixture.csv."""

import pytest
import json
import numpy as np
import pandas as pd

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


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
    tools = make_patterns_tools(session)
    return session, {t.name: t for t in tools}


def test_make_tools_returns_mine_patterns_tool():
    """Verify factory returns registered mine_patterns_tool."""
    session = SessionState()
    tools = make_patterns_tools(session)
    assert len(tools) == 1
    assert tools[0].name == "mine_patterns_tool"


def test_mine_patterns_tool_fpgrowth_known_answer_17():
    """Test mine_patterns_tool with FPGrowth against Known Answer 17 in TEST_FIXTURE.md."""
    session, tools = _load_fixture_dataset()
    mine_tool = tools["mine_patterns_tool"]

    # 1. catA with filtering_method='tfidf', algorithm='fpgrowth', min_sup=1
    res_catA = mine_tool.invoke(
        {
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": 1,
        }
    )
    assert res_catA["category_name"] == "catA"
    assert res_catA["filtering_method"] == "tfidf"
    assert res_catA["algorithm"] == "fpgrowth"
    assert res_catA["kept_terms"] == ["alpha"]
    assert res_catA["n_patterns"] == 1
    assert res_catA["patterns"] == [{"pattern": "alpha", "support": 3}]
    assert isinstance(json.dumps(res_catA), str)
    assert session.artifacts["patterns_catA_tfidf_fpgrowth"] == [{"pattern": "alpha", "support": 3}]

    # catA at min_sup=4 yields 0 patterns (Known Answer 17: support 3 is below threshold 4)
    res_catA_sup4 = mine_tool.invoke(
        {
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": 4,
        }
    )
    assert res_catA_sup4["n_patterns"] == 0
    assert res_catA_sup4["patterns"] == []

    # 2. catB with filtering_method='tfidf', algorithm='fpgrowth', min_sup=1
    res_catB = mine_tool.invoke(
        {
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": 1,
        }
    )
    assert res_catB["category_name"] == "catB"
    assert res_catB["filtering_method"] == "tfidf"
    assert res_catB["algorithm"] == "fpgrowth"
    assert res_catB["kept_terms"] == ["gamma"]
    assert res_catB["n_patterns"] == 1
    assert res_catB["patterns"] == [{"pattern": "gamma", "support": 4}]
    assert isinstance(json.dumps(res_catB), str)
    assert session.artifacts["patterns_catB_tfidf_fpgrowth"] == [{"pattern": "gamma", "support": 4}]

    # catB at min_sup=4 retains gamma with support 4
    res_catB_sup4 = mine_tool.invoke(
        {
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": 4,
        }
    )
    assert res_catB_sup4["n_patterns"] == 1
    assert res_catB_sup4["patterns"] == [{"pattern": "gamma", "support": 4}]


def test_mine_patterns_tool_topk_and_maxfpgrowth_known_answer_20():
    """Test mine_patterns_tool Top-K and MaxFPGrowth against Known Answer 20 in TEST_FIXTURE.md."""
    session, tools = _load_fixture_dataset()
    mine_tool = tools["mine_patterns_tool"]

    # 1. Top-K algorithm (k=1, k=2, k=3) for catA -> returns {alpha} (support 3)
    for k_val in [1, 2, 3]:
        res_catA_topk = mine_tool.invoke(
            {
                "category_name": "catA",
                "filtering_method": "tfidf",
                "algorithm": "topk",
                "k": k_val,
            }
        )
        assert res_catA_topk["n_patterns"] == 1
        assert res_catA_topk["patterns"] == [{"pattern": "alpha", "support": 3}]

    # Top-K algorithm (k=1, k=2, k=3) for catB -> returns {gamma} (support 4)
    for k_val in [1, 2, 3]:
        res_catB_topk = mine_tool.invoke(
            {
                "category_name": "catB",
                "filtering_method": "tfidf",
                "algorithm": "topk",
                "k": k_val,
            }
        )
        assert res_catB_topk["n_patterns"] == 1
        assert res_catB_topk["patterns"] == [{"pattern": "gamma", "support": 4}]

    # 2. MaxFPGrowth algorithm -> returns 0 patterns for both categories at min_sup=1 and min_sup=4
    # (Known Answer 20: PAMI Maximal FPGrowth returns empty result on single-term dataset, expected behavior)
    for sup in [1, 4]:
        res_catA_max = mine_tool.invoke(
            {
                "category_name": "catA",
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": sup,
            }
        )
        assert res_catA_max["n_patterns"] == 0
        assert res_catA_max["patterns"] == []

        res_catB_max = mine_tool.invoke(
            {
                "category_name": "catB",
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": sup,
            }
        )
        assert res_catB_max["n_patterns"] == 0
        assert res_catB_max["patterns"] == []


def test_mine_patterns_tool_zero_terms_survived():
    """Test variance and term_frequency filtering leaving zero terms for both categories."""
    session, tools = _load_fixture_dataset()
    mine_tool = tools["mine_patterns_tool"]

    # Known Answer 17: variance filtering keeps zero terms for both catA and catB
    for cat in ["catA", "catB"]:
        res_var = mine_tool.invoke(
            {
                "category_name": cat,
                "filtering_method": "variance",
                "algorithm": "fpgrowth",
                "min_sup": 1,
            }
        )
        assert res_var["n_patterns"] == 0
        assert res_var["kept_terms"] == []
        assert res_var["patterns"] == []
        assert "no terms survived filtering" in res_var["note"]

    # Known Answer 17: term_frequency filtering keeps zero terms for both catA and catB
    for cat in ["catA", "catB"]:
        res_tf = mine_tool.invoke(
            {
                "category_name": cat,
                "filtering_method": "term_frequency",
                "algorithm": "fpgrowth",
                "min_sup": 1,
            }
        )
        assert res_tf["n_patterns"] == 0
        assert res_tf["kept_terms"] == []
        assert res_tf["patterns"] == []
        assert "no terms survived filtering" in res_tf["note"]


def test_mine_patterns_tool_tfidf_and_empty_document_preservation():
    """Verify TF-IDF filtering retains alpha for catA and gamma for catB while preserving empty-text document."""
    session, tools = _load_fixture_dataset()
    mine_tool = tools["mine_patterns_tool"]

    # catA has 4 documents total, including document 7 which has empty text.
    # TF-IDF averaging over all 4 documents yields mean TF-IDF: alpha = 0.6376, beta = 0.3676.
    # 20th percentile threshold keeps only alpha.
    res_catA = mine_tool.invoke(
        {
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": 1,
        }
    )
    assert res_catA["kept_terms"] == ["alpha"]

    # catB mean TF-IDF: gamma = 0.8143, delta = 0.5444 -> keeps only gamma
    res_catB = mine_tool.invoke(
        {
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": 1,
        }
    )
    assert res_catB["kept_terms"] == ["gamma"]


def test_mine_patterns_tool_parameter_validation_and_errors():
    """Test early parameter validation errors and missing data handling."""
    session, tools = _load_fixture_dataset()
    mine_tool = tools["mine_patterns_tool"]

    # 1. Invalid filtering_method
    with pytest.raises(ValueError, match="Unsupported filtering_method 'invalid'"):
        mine_tool.invoke({"category_name": "catA", "filtering_method": "invalid", "algorithm": "fpgrowth", "min_sup": 1})

    # 2. Invalid algorithm
    with pytest.raises(ValueError, match="Unsupported algorithm 'invalid'"):
        mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "invalid", "min_sup": 1})

    # 3. Missing min_sup for fpgrowth
    with pytest.raises(ValueError, match="min_sup must be a positive integer"):
        mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth"})

    # 4. Invalid min_sup (< 1) for fpgrowth
    with pytest.raises(ValueError, match="min_sup must be a positive integer"):
        mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 0})

    # 5. Missing k for topk
    with pytest.raises(ValueError, match="k must be a positive integer"):
        mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk"})

    # 6. Invalid k (< 1) for topk
    with pytest.raises(ValueError, match="k must be a positive integer"):
        mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": -1})

    # 7. Non-existent category_name
    with pytest.raises(ValueError, match="Category 'non_existent' not found"):
        mine_tool.invoke({"category_name": "non_existent", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1})

    # 8. Missing text column
    session.dataframe = session.dataframe.drop(columns=["text"])
    with pytest.raises(ValueError, match="Column 'text' not found"):
        mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1})


def test_mine_patterns_tool_empty_vocabulary():
    """Test mine_patterns_tool when documents contain no valid english vocabulary words."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset where text contains only English stopwords ('the', 'and', 'is')
    df_stopwords = pd.DataFrame({"text": ["the the and", "is is the"], "label": ["catA", "catA"]})
    session.dataframe = df_stopwords
    session.set_labels(df_stopwords["label"].values)

    tools = {t.name: t for t in make_patterns_tools(session)}
    mine_tool = tools["mine_patterns_tool"]

    result = mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1})

    assert result["n_patterns"] == 0
    assert result["kept_terms"] == []
    assert result["patterns"] == []
    assert "no terms survived vectorization" in result["note"]


def test_mine_patterns_tool_uninitialized():
    """Test calling mine_patterns_tool on empty or uninitialized session."""
    session = SessionState()
    tools = {t.name: t for t in make_patterns_tools(session)}

    with pytest.raises(ValueError, match="session.dataframe is empty or None"):
        tools["mine_patterns_tool"].invoke({"category_name": "catA", "min_sup": 1})
