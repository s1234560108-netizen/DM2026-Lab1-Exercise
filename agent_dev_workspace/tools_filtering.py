"""Feature filtering tools based on term variance and statistical metrics."""

from typing import Union
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import rankdata, spearmanr
from langchain_core.tools import tool


def make_tools(session):
    """Factory function that returns feature filtering tools initialized with shared session state."""

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Filters Document-Term Matrix terms by variance cutoff.

        Calculates population variance (ddof=0) for each term across all documents.
        Retains terms with variance > threshold. Stores a full_report DataFrame
        for visualization tools.

        Args:
            threshold: Minimum variance threshold. Terms with variance > threshold are kept.

        Returns:
            dict: Summary containing result_id, threshold, n_terms_before,
                  n_terms_kept, n_terms_removed, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "session.feature_matrix or session.feature_names is None. "
                "Run build_dtm_tool first."
            )

        if threshold < 0.0:
            raise ValueError("threshold must be a non-negative float (>= 0.0).")

        X = session.feature_matrix
        feature_names = session.feature_names
        n_terms_before = len(feature_names)

        # Handle zero-feature or empty DTM matrices
        if n_terms_before == 0 or X.shape[1] == 0:
            result_id = session.next_result_id("variance_filter")
            full_report = pd.DataFrame(columns=["term", "variance"])
            summary = {
                "result_id": result_id,
                "threshold": threshold,
                "n_terms_before": 0,
                "n_terms_kept": 0,
                "n_terms_removed": 0,
                "kept_terms": [],
                "removed_terms": [],
            }
            session.store_result(
                "variance_filter_tool",
                {"threshold": threshold},
                summary,
                full_report=full_report,
            )
            return summary

        # Sparse-compatible population variance calculation (ddof=0)
        # E[X^2] - (E[X])^2
        n_docs = X.shape[0]
        if n_docs > 0:
            if issparse(X):
                mean = np.array(X.mean(axis=0)).ravel()
                # Square element-wise in sparse format
                X_squared = X.copy()
                X_squared.data **= 2
                mean_sq = np.array(X_squared.mean(axis=0)).ravel()
                variances = mean_sq - (mean ** 2)
                # Numerical stability cleanup (clip negative floating point noise)
                variances = np.maximum(variances, 0.0)
            else:
                variances = np.var(X, axis=0, ddof=0)
        else:
            variances = np.zeros(n_terms_before)

        # Filter terms strictly > threshold
        kept_mask = variances > threshold
        kept_terms = [feature_names[i] for i in range(n_terms_before) if kept_mask[i]]
        removed_terms = [feature_names[i] for i in range(n_terms_before) if not kept_mask[i]]

        # Construct full_report DataFrame: 1st column "term", 2nd column "variance", sorted descending
        full_report = pd.DataFrame(
            {
                "term": feature_names,
                "variance": variances,
            }
        ).sort_values(by="variance", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "n_terms_before": n_terms_before,
            "n_terms_kept": len(kept_terms),
            "n_terms_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=full_report,
        )
        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Filters DTM terms by Pearson correlation with a target class (one-vs-rest).

        Calculates Pearson correlation r between each term's count and a binary target
        vector (1 if target_class, 0 otherwise). Retains terms with absolute correlation
        |r| > threshold. Stores full_report DataFrame sorted by pearson_r in descending order.

        Args:
            target_class: Target category label string for one-vs-rest correlation.
            threshold: Minimum absolute Pearson correlation threshold (|r| > threshold).

        Returns:
            dict: Summary containing result_id, target_class, threshold, n_terms_before,
                  n_terms_kept, n_terms_removed, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "session.feature_matrix or session.feature_names is None. "
                "Run build_dtm_tool first."
            )

        if threshold < 0.0:
            raise ValueError("threshold must be a non-negative float (>= 0.0).")

        # Resolve labels
        labels = None
        if session.dataframe is not None:
            for col in ["category_name", "category", "label"]:
                if col in session.dataframe.columns:
                    labels = session.dataframe[col].values
                    break
        if labels is None and session.labels is not None:
            labels = session.labels

        if labels is None:
            raise ValueError("No category labels found in session. Load dataset with labels first.")

        # Ensure string comparison for target_class
        y = (np.array([str(lbl) for lbl in labels]) == str(target_class)).astype(float)
        if np.sum(y) == 0:
            raise ValueError(f"target_class '{target_class}' not found in dataset labels.")

        X = session.feature_matrix
        feature_names = session.feature_names
        n_terms_before = len(feature_names)

        if n_terms_before == 0 or X.shape[1] == 0:
            result_id = session.next_result_id("pearson_filter")
            full_report = pd.DataFrame(columns=["term", "pearson_r"])
            summary = {
                "result_id": result_id,
                "target_class": target_class,
                "threshold": threshold,
                "n_terms_before": 0,
                "n_terms_kept": 0,
                "n_terms_removed": 0,
                "kept_terms": [],
                "removed_terms": [],
            }
            session.store_result(
                "pearson_filter_tool",
                {"target_class": target_class, "threshold": threshold},
                summary,
                full_report=full_report,
            )
            return summary

        # Vectorized sparse-compatible Pearson r calculation
        # r = Cov(X, y) / (std(X) * std(y))
        n_docs = X.shape[0]
        y_centered = y - np.mean(y)
        y_std = np.std(y, ddof=0)

        if y_std == 0:
            # target vector is constant (e.g. all 1s or all 0s)
            pearson_r = np.zeros(n_terms_before)
        else:
            if issparse(X):
                x_mean = np.array(X.mean(axis=0)).ravel()
                # Compute X_centered * y_centered = (X * y_centered) - x_mean * sum(y_centered)
                # Since sum(y_centered) == 0, cov_xy = (X.T @ y_centered) / n_docs
                cov_xy = np.array(X.T.dot(y_centered)).ravel() / n_docs

                X_squared = X.copy()
                X_squared.data **= 2
                x_sq_mean = np.array(X_squared.mean(axis=0)).ravel()
                x_var = np.maximum(x_sq_mean - (x_mean ** 2), 0.0)
                x_std = np.sqrt(x_var)
            else:
                cov_xy = np.mean((X - np.mean(X, axis=0)) * y_centered[:, None], axis=0)
                x_std = np.std(X, axis=0, ddof=0)

            # Avoid division by zero for constant terms (x_std == 0)
            denominator = x_std * y_std
            pearson_r = np.where(denominator > 0, cov_xy / denominator, 0.0)
            # Clip numerical precision artifacts to [-1.0, 1.0]
            pearson_r = np.clip(pearson_r, -1.0, 1.0)

        # Filter using absolute correlation |r| > threshold
        kept_mask = np.abs(pearson_r) > threshold
        kept_terms = [feature_names[i] for i in range(n_terms_before) if kept_mask[i]]
        removed_terms = [feature_names[i] for i in range(n_terms_before) if not kept_mask[i]]

        # Construct full_report DataFrame sorted by signed pearson_r in descending order
        full_report = pd.DataFrame(
            {
                "term": feature_names,
                "pearson_r": pearson_r,
            }
        ).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "n_terms_before": n_terms_before,
            "n_terms_kept": len(kept_terms),
            "n_terms_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=full_report,
        )
        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Filters DTM terms by Spearman rank correlation with a target class (one-vs-rest).

        Calculates Spearman rank correlation rho between each term's count ranks and a binary target
        vector's ranks (1 if target_class, 0 otherwise). Retains terms with absolute correlation
        |rho| > threshold. Stores full_report DataFrame sorted by spearman_r in descending order.

        Args:
            target_class: Target category label string for one-vs-rest correlation.
            threshold: Minimum absolute Spearman correlation threshold (|rho| > threshold).

        Returns:
            dict: Summary containing result_id, target_class, threshold, n_terms_before,
                  n_terms_kept, n_terms_removed, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "session.feature_matrix or session.feature_names is None. "
                "Run build_dtm_tool first."
            )

        if threshold < 0.0:
            raise ValueError("threshold must be a non-negative float (>= 0.0).")

        # Resolve labels
        labels = None
        if session.dataframe is not None:
            for col in ["category_name", "category", "label"]:
                if col in session.dataframe.columns:
                    labels = session.dataframe[col].values
                    break
        if labels is None and session.labels is not None:
            labels = session.labels

        if labels is None:
            raise ValueError("No category labels found in session. Load dataset with labels first.")

        y = (np.array([str(lbl) for lbl in labels]) == str(target_class)).astype(float)
        if np.sum(y) == 0:
            raise ValueError(f"target_class '{target_class}' not found in dataset labels.")

        X = session.feature_matrix
        feature_names = session.feature_names
        n_terms_before = len(feature_names)

        if n_terms_before == 0 or X.shape[1] == 0:
            result_id = session.next_result_id("spearman_filter")
            full_report = pd.DataFrame(columns=["term", "spearman_r"])
            summary = {
                "result_id": result_id,
                "target_class": target_class,
                "threshold": threshold,
                "n_terms_before": 0,
                "n_terms_kept": 0,
                "n_terms_removed": 0,
                "kept_terms": [],
                "removed_terms": [],
            }
            session.store_result(
                "spearman_filter_tool",
                {"target_class": target_class, "threshold": threshold},
                summary,
                full_report=full_report,
            )
            return summary

        # Compute Spearman rank correlation rho
        # Rank target vector y with average ties
        y_ranked = rankdata(y)

        # Convert CSC matrix for efficient per-column ranking and Spearman calculations
        X_csc = X.tocsc() if issparse(X) else X
        spearman_rhos = np.zeros(n_terms_before)

        for col_idx in range(n_terms_before):
            if issparse(X_csc):
                col_data = X_csc[:, col_idx].toarray().ravel()
            else:
                col_data = X_csc[:, col_idx]

            # Calculate Spearman rho handling ties cleanly via scipy.stats.spearmanr
            rho, _ = spearmanr(col_data, y_ranked)
            if np.isnan(rho):
                rho = 0.0
            spearman_rhos[col_idx] = float(np.clip(rho, -1.0, 1.0))

        # Filter using absolute rank correlation |rho| > threshold
        kept_mask = np.abs(spearman_rhos) > threshold
        kept_terms = [feature_names[i] for i in range(n_terms_before) if kept_mask[i]]
        removed_terms = [feature_names[i] for i in range(n_terms_before) if not kept_mask[i]]

        # Construct full_report DataFrame sorted by signed spearman_r in descending order
        full_report = pd.DataFrame(
            {
                "term": feature_names,
                "spearman_r": spearman_rhos,
            }
        ).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "n_terms_before": n_terms_before,
            "n_terms_kept": len(kept_terms),
            "n_terms_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=full_report,
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
