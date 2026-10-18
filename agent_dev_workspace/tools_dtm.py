"""Document-Term Matrix (DTM) tools for text vectorization and representation."""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    """Factory function that returns DTM tools initialized with shared session state."""

    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: Optional[List[int]] = None,
    ) -> dict:
        """Builds a Document-Term Matrix (DTM) from text in session.dataframe.

        Extracts word/ngram frequencies using CountVectorizer, stores the resulting
        sparse matrix and feature names in the session, and computes matrix sparsity.

        Args:
            max_features: Maximum number of terms to keep in vocabulary, sorted by term frequency.
            min_df: Minimum document frequency for terms (int count or float proportion).
            max_df: Maximum document frequency for terms (int count or float proportion).
            ngram_range: Range of n-gram sizes as [min_n, max_n], e.g., [1, 1] for unigrams.

        Returns:
            dict: Summary containing result_id, n_documents, vocabulary_size,
                  feature_names, non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("session.dataframe is empty or None. Load a dataset first.")

        # Validate ngram_range
        if ngram_range is None:
            ngram_range = [1, 1]
        elif (
            not isinstance(ngram_range, (list, tuple))
            or len(ngram_range) != 2
            or not all(isinstance(x, int) and x >= 1 for x in ngram_range)
            or ngram_range[0] > ngram_range[1]
        ):
            raise ValueError(
                "ngram_range must be a sequence of 2 positive integers [min_n, max_n] with min_n <= max_n"
            )

        ngram_tuple = (ngram_range[0], ngram_range[1])

        # Fill missing text values with empty string to preserve document row count
        texts = session.dataframe["text"].fillna("")
        n_documents = len(texts)

        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
        )

        try:
            X = vectorizer.fit_transform(texts)
            feature_names = [str(name) for name in vectorizer.get_feature_names_out()]
            fitted_vectorizer = vectorizer
        except ValueError as e:
            err_msg = str(e).lower()
            # Catch empty vocabulary or pruning elimination errors specifically, re-raise unrelated ValueErrors
            if "empty vocabulary" in err_msg or "no terms remain" in err_msg:
                X = csr_matrix((n_documents, 0), dtype=np.int64)
                feature_names = []
                fitted_vectorizer = None
            else:
                raise

        # Store matrix, feature names, and vectorizer in session
        session.feature_matrix = X
        session.feature_names = feature_names
        if fitted_vectorizer is not None:
            session.artifacts["count_vectorizer"] = fitted_vectorizer
        else:
            session.artifacts.pop("count_vectorizer", None)

        # Sync labels if session.labels is not already initialized
        if session.labels is None:
            for col in ["category_name", "category", "label"]:
                if col in session.dataframe.columns:
                    session.set_labels(session.dataframe[col].values)
                    break

        vocabulary_size = len(feature_names)
        non_zero = int(X.nnz)
        total_elements = n_documents * vocabulary_size

        if total_elements > 0:
            sparsity_pct = 100.0 * (1.0 - (non_zero / total_elements))
        else:
            sparsity_pct = 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "vocabulary_size": vocabulary_size,
            "feature_names": feature_names,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result(
            "build_dtm_tool",
            {
                "max_features": max_features,
                "min_df": min_df,
                "max_df": max_df,
                "ngram_range": ngram_range,
            },
            summary,
        )
        return summary

    @tool
    def term_frequency_tool() -> dict:
        """Aggregates term frequencies (column sums) across all documents in the DTM.

        Returns:
            dict: Summary containing result_id, vocabulary_size, term_frequencies dict,
                  and top_terms preview.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "session.feature_matrix or session.feature_names is None. "
                "Run build_dtm_tool first."
            )

        X = session.feature_matrix
        feature_names = [str(f) for f in session.feature_names]
        vocabulary_size = len(feature_names)

        if vocabulary_size == 0 or X.shape[1] == 0:
            result_id = session.next_result_id("term_frequency")
            full_report = pd.DataFrame(columns=["term", "frequency"])
            summary = {
                "result_id": result_id,
                "vocabulary_size": 0,
                "term_frequencies": {},
                "top_terms": {},
            }
            session.store_result(
                "term_frequency_tool",
                {},
                summary,
                full_report=full_report,
            )
            return summary

        # Sum total counts per column across all documents
        counts = np.array(X.sum(axis=0)).ravel()
        term_freqs = {feature_names[i]: int(counts[i]) for i in range(vocabulary_size)}

        # Full report DataFrame for store_result / visualize_result_tool
        full_report = pd.DataFrame(
            {"term": feature_names, "frequency": [int(c) for c in counts]}
        ).sort_values(by="frequency", ascending=False).reset_index(drop=True)

        # Build preview: top 20 terms
        n_preview = min(vocabulary_size, 20)
        top_terms = dict(zip(full_report["term"].head(n_preview), full_report["frequency"].head(n_preview).astype(int)))

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "vocabulary_size": vocabulary_size,
            "term_frequencies": term_freqs if vocabulary_size <= 20 else top_terms,
            "top_terms": top_terms,
        }

        session.store_result("term_frequency_tool", {}, summary, full_report=full_report)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Generates a heatmap visualization of a raw positional slice of the global DTM.

        Slices the matrix in existing row (document) and column (vocabulary) order
        without reordering terms by frequency or variance. Sets session.pending_figure.

        Args:
            n_terms: Number of terms (columns) to slice from the start of the vocabulary (default 20).
            n_documents: Number of documents (rows) to slice from the start of the dataset (default 20).

        Returns:
            dict: Summary containing result_id, sliced_documents, sliced_terms,
                  terms_included, and matrix values.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "session.feature_matrix or session.feature_names is None. "
                "Run build_dtm_tool first."
            )

        if n_terms < 1 or n_documents < 1:
            raise ValueError("n_terms and n_documents must be positive integers (>= 1).")

        X = session.feature_matrix
        feature_names = [str(f) for f in session.feature_names]
        total_docs, total_terms = X.shape

        # Slice actual dimensions bounded by available matrix size
        slice_docs = min(n_documents, total_docs)
        slice_terms = min(n_terms, total_terms)

        sliced_terms = feature_names[:slice_terms]

        if slice_docs == 0 or slice_terms == 0:
            result_id = session.next_result_id("dtm_heatmap")
            summary = {
                "result_id": result_id,
                "sliced_documents": 0,
                "sliced_terms": 0,
                "terms_included": [],
                "matrix": [],
            }
            session.store_result(
                "dtm_heatmap_tool",
                {"n_terms": n_terms, "n_documents": n_documents},
                summary,
            )
            return summary

        # Positional slicing of raw counts
        X_slice = X[:slice_docs, :slice_terms]
        matrix_dense = X_slice.toarray() if issparse(X_slice) else np.asarray(X_slice)
        # Ensure native Python int conversion for JSON-serialization
        matrix_list = [[int(val) for val in row] for row in matrix_dense]

        # Build Matplotlib / Seaborn heatmap figure
        fig, ax = plt.subplots(figsize=(max(6, slice_terms * 0.8), max(4, slice_docs * 0.5)))
        sns.heatmap(
            matrix_dense,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=sliced_terms,
            yticklabels=[f"Doc {i}" for i in range(slice_docs)],
            ax=ax,
            cbar=True,
        )
        ax.set_title(f"DTM Heatmap (Top {slice_docs} Docs x Top {slice_terms} Terms)")
        ax.set_xlabel("Vocabulary Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Set figure on session.pending_figure (no savefig or show)
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "sliced_documents": slice_docs,
            "sliced_terms": slice_terms,
            "terms_included": sliced_terms,
            "matrix": matrix_list,
        }

        full_report = pd.DataFrame(matrix_dense, columns=sliced_terms)
        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
            full_report=full_report,
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
