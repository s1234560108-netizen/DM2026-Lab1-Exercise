"""Frequent-pattern mining tools using PAMI algorithms on category-specific document-term matrices."""

import tempfile
from pathlib import Path
from typing import List, Optional, Literal
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

from PAMI.extras.convert.DF2DB import DF2DB
from PAMI.frequentPattern.basic.FPGrowth import FPGrowth
from PAMI.frequentPattern.topk.FAE import FAE
from PAMI.frequentPattern.maximal.MaxFPGrowth import MaxFPGrowth


def make_tools(session):
    """Factory function that returns pattern mining tools initialized with shared session state."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> dict:
        """Mines frequent word patterns for a specific document category using PAMI algorithms.

        Builds a fresh, category-local document-term matrix with CountVectorizer(stop_words='english'),
        applies category-local vocabulary filtering ('variance', 'tfidf', or 'term_frequency'),
        converts surviving terms to a transactional database, and mines frequent patterns
        using FPGrowth, Top-K (FAE), or MaxFPGrowth.

        Args:
            category_name: Target category string to mine (e.g. 'catA', 'catB').
            filtering_method: Vocabulary filter ('tfidf', 'variance', or 'term_frequency'). Default 'tfidf'.
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Default 'fpgrowth'.
            min_sup: Integer minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Integer top-K pattern count (required for 'topk').

        Returns:
            dict: Summary containing result_id, category_name, filtering_method, algorithm,
                  kept_terms, n_patterns, and patterns list.
        """
        # 1. Early input validation
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("session.dataframe is empty or None. Load a dataset first.")

        filter_clean = str(filtering_method).lower().strip()
        if filter_clean not in ["variance", "tfidf", "term_frequency"]:
            raise ValueError(f"Unsupported filtering_method '{filtering_method}'. Must be 'variance', 'tfidf', or 'term_frequency'.")

        alg_clean = str(algorithm).lower().strip()
        if alg_clean in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or not isinstance(min_sup, int) or min_sup < 1:
                raise ValueError(f"min_sup must be a positive integer for algorithm '{algorithm}'.")
        elif alg_clean == "topk":
            if k is None or not isinstance(k, int) or k < 1:
                raise ValueError(f"k must be a positive integer for algorithm '{algorithm}'.")
        else:
            raise ValueError(f"Unsupported algorithm '{algorithm}'. Must be 'fpgrowth', 'topk', or 'maxfpgrowth'.")

        df = session.dataframe

        # Validate standardized text column
        if "text" not in df.columns:
            raise ValueError("Column 'text' not found in session.dataframe.")

        # Determine category column and filter rows while preserving all category documents
        cat_col = None
        for col in ["category_name", "label", "category"]:
            if col in df.columns:
                if col == "category" and "category_name" in df.columns:
                    continue
                cat_col = col
                break

        if cat_col is None:
            raise ValueError("No category column found in session.dataframe.")

        category_mask = df[cat_col].astype(str) == str(category_name)
        category_df = df[category_mask].copy()

        if category_df.empty:
            raise ValueError(f"Category '{category_name}' not found in session.dataframe.")

        # Fill missing/empty text with empty string "" to preserve category documents in vectorization and TF-IDF
        texts = category_df["text"].fillna("").astype(str).tolist()

        # 2. Build fresh category-local CountVectorizer(stop_words="english")
        vectorizer = CountVectorizer(stop_words="english")
        try:
            counts_matrix = vectorizer.fit_transform(texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError as err:
            err_msg = str(err).lower()
            if "empty vocabulary" in err_msg or "after stop words removal" in err_msg:
                feature_names = np.array([])
                counts_matrix = None
            else:
                raise

        if feature_names.size == 0 or counts_matrix is None:
            result_id = session.next_result_id(f"patterns_{category_name}")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filter_clean,
                "algorithm": alg_clean,
                "kept_terms": [],
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived vectorization",
            }
            session.artifacts[f"patterns_{category_name}_{filter_clean}_{alg_clean}"] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        dtm_df = pd.DataFrame(counts_matrix.toarray(), columns=feature_names)

        # 3. Apply category-local filtering method using exact percentile rules
        if filter_clean == "variance":
            metric = dtm_df.var(axis=0)
            lower = np.percentile(metric, 5)
            upper = np.percentile(metric, 95)
            mask = (metric >= lower) & (metric <= upper)
            kept_terms = dtm_df.columns[mask].tolist()
        elif filter_clean == "term_frequency":
            metric = dtm_df.sum(axis=0)
            lower = np.percentile(metric, 5)
            upper = np.percentile(metric, 95)
            mask = (metric >= lower) & (metric <= upper)
            kept_terms = dtm_df.columns[mask].tolist()
        elif filter_clean == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(dtm_df.values)
            scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            threshold = np.percentile(scores, 20)
            mask = scores >= threshold
            kept_terms = dtm_df.columns[mask].tolist()

        # Handle zero terms surviving filtering
        if not kept_terms:
            result_id = session.next_result_id(f"patterns_{category_name}")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filter_clean,
                "algorithm": alg_clean,
                "kept_terms": [],
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.artifacts[f"patterns_{category_name}_{filter_clean}_{alg_clean}"] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        filtered_df = dtm_df[kept_terms]

        # 4. Convert to transactional database and mine with PAMI inside temp directory
        patterns = []
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_db_path = str(Path(temp_dir) / "category_transactions.db")

            converter = DF2DB(filtered_df, DFtype="dense")
            converter.convert2TransactionalDatabase(temp_db_path, condition=">=", thresholdValue=1)

            if alg_clean == "fpgrowth":
                miner = FPGrowth(temp_db_path, minSup=min_sup, sep="\t")
            elif alg_clean == "topk":
                miner = FAE(temp_db_path, k=k, sep="\t")
            elif alg_clean == "maxfpgrowth":
                miner = MaxFPGrowth(temp_db_path, minSup=min_sup, sep="\t")

            miner.mine()
            patterns_df = miner.getPatternsAsDataFrame()

            if patterns_df is not None and not patterns_df.empty:
                pattern_col = patterns_df.columns[0]
                support_col = patterns_df.columns[1]

                for _, row in patterns_df.iterrows():
                    p_str = str(row[pattern_col]).strip()
                    s_val = int(row[support_col])
                    patterns.append({"pattern": p_str, "support": s_val})

        # 5. Store artifacts and result summary
        artifact_key = f"patterns_{category_name}_{filter_clean}_{alg_clean}"
        session.artifacts[artifact_key] = patterns

        result_id = session.next_result_id(f"patterns_{category_name}")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filter_clean,
            "algorithm": alg_clean,
            "kept_terms": kept_terms,
            "n_patterns": len(patterns),
            "patterns": patterns,
        }

        session.store_result(
            "mine_patterns_tool",
            {
                "category_name": category_name,
                "filtering_method": filter_clean,
                "algorithm": alg_clean,
                "min_sup": min_sup,
                "k": k,
            },
            summary,
        )
        return summary

    return [mine_patterns_tool]
