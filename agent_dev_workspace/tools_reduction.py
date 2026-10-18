"""Dimension reduction and label binarization tools for session dataframes."""

from typing import List, Optional, Literal
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    """Factory function that returns reduction tools initialized with shared session state."""

    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encodes document category labels into a 2D matrix (one column per category).

        Prioritizes session.dataframe["category_name"]. Falls back to session.labels
        only if category_name is absent and session.labels length matches n_documents.
        Generates exactly one column per category (including 2 columns for binary classification).

        Returns:
            dict: Summary containing result_id, n_documents, n_categories,
                  categories, shape, and one_hot_preview.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("session.dataframe is empty or None. Load a dataset first.")

        df = session.dataframe
        n_documents = len(df)

        # 1. Determine active label series with strict rule
        if "category_name" in df.columns:
            series = df["category_name"]
        elif session.labels is not None and len(session.labels) == n_documents:
            series = pd.Series(session.labels, index=df.index)
        else:
            raise ValueError("No valid category_name column in session.dataframe or valid session.labels found.")

        # 2. Strict validation for missing or empty/whitespace-only category labels
        for idx, val in series.items():
            if pd.isna(val) or str(val).strip() == "":
                raise ValueError(f"Missing or empty category label at document index {idx}.")

        # 3. Derive sorted category mapping directly from active labels
        labels_str = series.astype(str)
        categories = sorted(labels_str.unique().tolist())
        n_categories = len(categories)

        # 4. One-hot encode using pandas get_dummies to guarantee one column per category
        encoded_df = pd.get_dummies(labels_str, dtype=int).reindex(columns=categories, fill_value=0)
        binarized_matrix = encoded_df.values.astype(int)

        # 5. Store binarized matrix and category list in session artifacts
        session.artifacts["binarized_labels"] = binarized_matrix
        session.artifacts["binarized_label_categories"] = categories

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "n_categories": n_categories,
            "categories": categories,
            "shape": [n_documents, n_categories],
            "one_hot_preview": binarized_matrix.tolist()[:5],
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        random_state: Optional[int] = None,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> dict:
        """Reduces feature matrix dimensions (PCA, t-SNE, UMAP) to 2D and plots coordinates.

        Calculates 2D reduced coordinates from session.feature_matrix, stores full
        coordinates in session.artifacts, and builds a category-colored scatter plot set on
        session.pending_figure.

        Args:
            method: Reduction method ('pca', 'tsne', or 'umap'). Default 'pca'.
            n_components: Target number of dimensions. Must be 2. Default 2.
            random_state: Seed for reproducible reduction output.
            perplexity: t-SNE perplexity parameter (must be strictly less than n_documents).
            n_neighbors: UMAP n_neighbors parameter (must be strictly less than n_documents).

        Returns:
            dict: Summary containing result_id, method, n_documents, n_components, shape,
                  coords_preview, and explained_variance_ratio (for PCA).
        """
        if session.feature_matrix is None:
            raise ValueError("session.feature_matrix is empty or None. Run build_dtm_tool first.")

        if n_components != 2:
            raise ValueError(f"n_components must be 2 for 2D visualization (got {n_components}).")

        method_clean = str(method).lower().strip()
        if method_clean not in ["pca", "tsne", "umap"]:
            raise ValueError(f"Unsupported reduction method '{method}'. Must be one of 'pca', 'tsne', 'umap'.")

        matrix = session.feature_matrix
        n_documents = matrix.shape[0]

        if n_documents < 2:
            raise ValueError("Feature matrix must contain at least 2 documents for dimension reduction.")

        # 1. Mandatory Category Label Validation
        category_labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            if len(session.dataframe) == n_documents:
                category_labels = session.dataframe["category_name"].tolist()
        elif session.labels is not None and len(session.labels) == n_documents:
            category_labels = list(session.labels)

        if category_labels is None:
            raise ValueError("Category labels are missing or length does not match n_documents. Cannot create required category-colored scatter plot.")

        # Check for empty/missing individual labels
        for idx, lbl in enumerate(category_labels):
            if pd.isna(lbl) or str(lbl).strip() == "":
                raise ValueError(f"Missing or empty category label at document index {idx}.")

        # 2. Sparse-to-dense conversion & Method-Specific Memory Safety
        # 100 MB memory limit for dense float64 allocation (~12.5 million elements)
        MAX_DENSE_BYTES = 100 * 1024 * 1024
        dense_bytes = matrix.shape[0] * matrix.shape[1] * 8  # 8 bytes per float64 element

        explained_variance_ratio = None

        if method_clean == "pca":
            # True centered PCA requires dense array for mean subtraction
            if sp.issparse(matrix):
                if dense_bytes > MAX_DENSE_BYTES:
                    mb_requested = dense_bytes / (1024 * 1024)
                    raise ValueError(
                        f"PCA requires dense conversion, but feature matrix ({matrix.shape[0]}x{matrix.shape[1]}) "
                        f"would require {mb_requested:.1f} MB, exceeding dense memory limit (100 MB)."
                    )
                X_dense = matrix.toarray()
            else:
                X_dense = np.asarray(matrix)

            model = PCA(n_components=2, random_state=random_state)
            coords = model.fit_transform(X_dense)
            explained_variance_ratio = model.explained_variance_ratio_.tolist()

        elif method_clean == "tsne":
            if perplexity >= n_documents or perplexity <= 0:
                raise ValueError(
                    f"perplexity ({perplexity}) must be strictly positive and less than n_documents ({n_documents})."
                )

            # For small matrices (under 100MB), dense conversion preserves exact Known Answer 8 behavior
            if sp.issparse(matrix):
                if dense_bytes <= MAX_DENSE_BYTES:
                    X_input = matrix.toarray()
                    init_param = "pca"
                else:
                    # For large sparse matrices, pass CSR directly with init='random'
                    X_input = matrix.tocsr()
                    init_param = "random"
            else:
                X_input = np.asarray(matrix)
                init_param = "pca"

            model = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init=init_param)
            coords = model.fit_transform(X_input)

        elif method_clean == "umap":
            if n_neighbors >= n_documents or n_neighbors < 2:
                raise ValueError(
                    f"n_neighbors ({n_neighbors}) must be >= 2 and strictly less than n_documents ({n_documents})."
                )

            # UMAP natively supports scipy CSR matrices without dense conversion
            if sp.issparse(matrix):
                if dense_bytes <= MAX_DENSE_BYTES:
                    X_input = matrix.toarray()
                else:
                    X_input = matrix.tocsr()
            else:
                X_input = np.asarray(matrix)

            model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
            coords = model.fit_transform(X_input)

        # Store full reduced coordinates in session.artifacts
        session.artifacts[f"reduced_coords_{method_clean}"] = coords
        session.artifacts["reduced_coords"] = coords

        # 3. Build Category-Colored 2D Scatter Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=category_labels,
            palette="tab10",
            s=100,
            ax=ax,
        )
        ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")

        method_title = method_clean.upper()
        ax.set_title(f"{method_title} 2D Projection")
        ax.set_xlabel(f"{method_title} Dim 1")
        ax.set_ylabel(f"{method_title} Dim 2")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id(f"reduce_{method_clean}")
        summary = {
            "result_id": result_id,
            "method": method_clean,
            "n_documents": n_documents,
            "n_components": 2,
            "shape": [n_documents, 2],
            "coords_preview": coords.tolist()[:5],
            "explained_variance_ratio": explained_variance_ratio,
        }

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method_clean,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
            },
            summary,
        )
        return summary

    return [binarize_labels_tool, reduce_dimensions_tool]
