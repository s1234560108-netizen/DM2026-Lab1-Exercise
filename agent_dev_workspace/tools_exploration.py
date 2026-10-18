"""Exploration tools for document similarity and feature correlation analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from langchain_core.tools import tool


def make_tools(session):
    """Factory function that returns exploration tools initialized with shared session state."""

    @tool
    def cosine_similarity_tool(doc_idx_1: int, doc_idx_2: int) -> dict:
        """Computes cosine similarity between two documents using session.feature_matrix.

        Args:
            doc_idx_1: Row index of the first document (0-indexed).
            doc_idx_2: Row index of the second document (0-indexed).

        Returns:
            dict: Summary containing result_id, doc_idx_1, doc_idx_2, and similarity score.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "session.feature_matrix or session.feature_names is None. "
                "Run build_dtm_tool first."
            )

        X = session.feature_matrix
        n_docs = X.shape[0]

        if doc_idx_1 < 0 or doc_idx_1 >= n_docs:
            raise ValueError(f"doc_idx_1 ({doc_idx_1}) out of bounds for dataset with {n_docs} documents.")
        if doc_idx_2 < 0 or doc_idx_2 >= n_docs:
            raise ValueError(f"doc_idx_2 ({doc_idx_2}) out of bounds for dataset with {n_docs} documents.")

        # Extract document vectors
        vec1 = X[doc_idx_1].toarray().ravel() if issparse(X) else np.asarray(X[doc_idx_1]).ravel()
        vec2 = X[doc_idx_2].toarray().ravel() if issparse(X) else np.asarray(X[doc_idx_2]).ravel()

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # Handle zero-vector documents safely (e.g., empty text row 7 vs anything = 0.0)
        if norm1 == 0.0 or norm2 == 0.0:
            similarity = 0.0
        else:
            dot_product = np.dot(vec1, vec2)
            similarity = float(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_idx_1": doc_idx_1,
            "doc_idx_2": doc_idx_2,
            "cosine_similarity": round(similarity, 4),
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_idx_1": doc_idx_1, "doc_idx_2": doc_idx_2},
            summary,
        )
        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """Computes Pearson correlation across top-20 terms by variance and generates a heatmap.

        Operates on global session.feature_matrix. Selects up to top 20 features by population
        variance, computes their correlation matrix, builds an annotated heatmap, and sets
        session.pending_figure before returning.

        Returns:
            dict: Summary containing result_id, n_terms, term_names, and correlation_matrix list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "session.feature_matrix or session.feature_names is None. "
                "Run build_dtm_tool first."
            )

        X = session.feature_matrix
        feature_names = [str(f) for f in session.feature_names]
        n_docs, n_terms = X.shape

        if n_terms == 0 or n_docs == 0:
            result_id = session.next_result_id("feature_correlation_matrix")
            summary = {
                "result_id": result_id,
                "n_terms": 0,
                "term_names": [],
                "correlation_matrix": [],
            }
            session.store_result("feature_correlation_matrix_tool", {}, summary)
            return summary

        # 1. Compute term population variances (ddof=0)
        if issparse(X):
            mean = np.array(X.mean(axis=0)).ravel()
            X_sq = X.copy()
            X_sq.data **= 2
            mean_sq = np.array(X_sq.mean(axis=0)).ravel()
            variances = np.maximum(mean_sq - (mean ** 2), 0.0)
        else:
            variances = np.var(X, axis=0, ddof=0)

        # 2. Take top 20 terms by variance descending
        k = min(20, n_terms)
        top_indices = np.argsort(variances)[::-1][:k]
        top_terms = [feature_names[idx] for idx in top_indices]

        # Extract top-variance matrix slice
        X_sub = X[:, top_indices]
        X_sub_dense = X_sub.toarray() if issparse(X_sub) else np.asarray(X_sub)

        # 3. Compute Pearson correlation matrix across top terms
        # Handle zero-variance terms safely
        std_sub = np.std(X_sub_dense, axis=0, ddof=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_matrix = np.corrcoef(X_sub_dense, rowvar=False)
            if k == 1:
                corr_matrix = np.array([[1.0]])
            else:
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                # Ensure diagonal is exactly 1.0
                np.fill_diagonal(corr_matrix, 1.0)
                corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        # Convert to native python floats for JSON serialization
        matrix_list = [[round(float(val), 4) for val in row] for row in corr_matrix]

        # 4. Build Annotated Heatmap Figure
        fig, ax = plt.subplots(figsize=(max(6, k * 0.8), max(5, k * 0.7)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
            cbar=True,
        )
        ax.set_title(f"Feature Correlation Matrix (Top {k} Terms by Variance)")
        plt.tight_layout()

        # Set figure on session.pending_figure (no savefig or show)
        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation_matrix")
        summary = {
            "result_id": result_id,
            "n_terms": k,
            "term_names": top_terms,
            "correlation_matrix": matrix_list,
        }

        full_report = pd.DataFrame(matrix_list, index=top_terms, columns=top_terms)
        session.store_result("feature_correlation_matrix_tool", {}, summary, full_report=full_report)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
