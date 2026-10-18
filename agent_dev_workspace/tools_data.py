"""Data quality, inspection, and sampling tools for session dataframes."""

import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from langchain_core.tools import tool


def _find_repo_root() -> Path:
    """Finds the root directory of the repository by walking up to find pyproject.toml or .git."""
    curr = Path.cwd().resolve()
    for parent in [curr] + list(curr.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return curr


def _truncate_preview_records(records: list) -> list:
    """Safely truncates long text and list fields in preview dictionaries for JSON serialization."""
    truncated_records = []
    for rec in records:
        clean_rec = {}
        for k, v in rec.items():
            if isinstance(v, str) and len(v) > 200:
                clean_rec[k] = v[:200] + "..."
            elif isinstance(v, list) and len(v) > 20:
                clean_rec[k] = v[:20] + ["..."]
            else:
                clean_rec[k] = v
        truncated_records.append(clean_rec)
    return truncated_records


def make_tools(session):
    """Factory function that returns data tools initialized with shared session state."""

    @tool
    def check_missing_tool(text_column: str = "text", drop: bool = False) -> dict:
        """Checks for missing or empty text values in session.dataframe.

        Identifies rows where text is null, NaN, or empty/whitespace-only.
        Optionally drops missing rows and updates session state cleanly.

        Args:
            text_column: Name of the text column to inspect (default 'text').
            drop: If True, drops missing text rows from session.dataframe.

        Returns:
            dict: Summary containing result_id, total_rows, missing_count,
                  missing_indices, and dropped.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("session.dataframe is empty or None. Load a dataset first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        df = session.dataframe
        total_rows = len(df)

        # Identify null, NaN, or empty/whitespace-only text
        series = df[text_column]
        missing_mask = series.isna() | (series.astype(str).str.strip() == "")
        missing_indices = df.index[missing_mask].tolist()
        missing_count = len(missing_indices)

        if drop and missing_count > 0:
            session.dataframe = df.drop(index=missing_indices).reset_index(drop=True)

            # Invalidate stale document-aligned matrices and artifacts
            session.feature_matrix = None
            session.feature_names = None
            session.artifacts.pop("count_vectorizer", None)
            session.artifacts.pop("unigrams", None)

            # Resync labels if present in remaining dataframe
            for col in ["category_name", "category", "label"]:
                if col in session.dataframe.columns:
                    session.set_labels(session.dataframe[col].values)
                    break

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_indices": missing_indices,
            "dropped": drop,
            "remaining_rows": len(session.dataframe),
        }

        session.store_result("check_missing_tool", {"text_column": text_column, "drop": drop}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Checks for duplicate document rows in session.dataframe.

        Identifies exact duplicate rows across all columns. When drop=True, removes
        duplicates using keep=False (removing every copy of a duplicated row),
        matching Master notebook behavior exactly.

        Args:
            drop: If True, drops all copies of duplicated rows from session.dataframe.

        Returns:
            dict: Summary containing result_id, total_rows, duplicate_occurrences,
                  duplicated_indices, dropped, and remaining_rows.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("session.dataframe is empty or None. Load a dataset first.")

        df = session.dataframe
        total_rows = len(df)

        # Count duplicate occurrences (rows that are repeated)
        dup_mask = df.duplicated(keep="first")
        duplicated_indices = df.index[dup_mask].tolist()
        duplicate_occurrences = len(duplicated_indices)

        if drop and duplicate_occurrences > 0:
            # Drop every copy of duplicated rows (keep=False per Master / Known Answer 3)
            session.dataframe = df.drop_duplicates(keep=False).reset_index(drop=True)

            # Invalidate stale document-aligned matrices and artifacts
            session.feature_matrix = None
            session.feature_names = None
            session.artifacts.pop("count_vectorizer", None)
            session.artifacts.pop("unigrams", None)

            # Resync labels if present in remaining dataframe
            for col in ["category_name", "category", "label"]:
                if col in session.dataframe.columns:
                    session.set_labels(session.dataframe[col].values)
                    break

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "duplicate_occurrences": duplicate_occurrences,
            "duplicated_indices": duplicated_indices,
            "dropped": drop,
            "remaining_rows": len(session.dataframe),
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> dict:
        """Lists files and directories within the repository or a specified subdirectory.

        Args:
            subdirectory: Relative path to a folder within the repository.

        Returns:
            dict: Summary containing result_id, subdirectory, directories, and files.
        """
        repo_root = _find_repo_root()
        target_dir = (repo_root / subdirectory).resolve()

        # Prevent directory traversal outside repo root using Path.is_relative_to
        try:
            if not target_dir.is_relative_to(repo_root):
                raise ValueError(f"Access denied: subdirectory '{subdirectory}' outside repository root.")
        except AttributeError:
            if not str(target_dir).startswith(str(repo_root)):
                raise ValueError(f"Access denied: subdirectory '{subdirectory}' outside repository root.")

        if not target_dir.exists() or not target_dir.is_dir():
            raise ValueError(f"Subdirectory '{subdirectory}' does not exist or is not a directory.")

        directories = []
        files = []

        for item in sorted(target_dir.iterdir()):
            if item.name.startswith("."):
                continue  # Skip hidden files/dirs
            if item.is_dir():
                directories.append(item.name)
            elif item.is_file():
                files.append(item.name)

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n_rows: int = 5) -> dict:
        """Inspects the first few rows and column structure of session.dataframe.

        Args:
            n_rows: Number of head rows to inspect (default 5, capped at 20).

        Returns:
            dict: Summary containing result_id, total_rows, columns, and preview dicts.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("session.dataframe is empty or None. Load a dataset first.")

        if n_rows < 1:
            raise ValueError("n_rows must be a positive integer (>= 1).")

        df = session.dataframe
        columns = list(df.columns)
        n_preview = min(n_rows, 20)
        head_records = df.head(n_preview).to_dict(orient="records")
        preview = _truncate_preview_records(head_records)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": len(df),
            "columns": columns,
            "preview_rows": len(head_records),
            "preview": preview,
        }

        session.store_result("inspect_data_tool", {"n_rows": n_rows}, summary)
        return summary

    @tool
    def tokenize_tool(text_column: str = "text") -> dict:
        """Tokenizes text into unigrams using NLTK word_tokenize on lowercased text.

        Args:
            text_column: Name of text column to tokenize (default 'text').

        Returns:
            dict: Summary containing result_id, total_documents, total_tokens, and a brief token preview.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("session.dataframe is empty or None. Load a dataset first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        # Ensure NLTK tokenizer resources are downloaded quietly if missing
        for resource in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
            try:
                nltk.data.find(resource)
            except LookupError:
                res_name = resource.split("/")[-1]
                nltk.download(res_name, quiet=True)

        tokenized_docs = []
        total_tokens = 0
        for val in session.dataframe[text_column]:
            if pd.isna(val) or str(val).strip() == "":
                tokenized_docs.append([])
            else:
                lowercased_text = str(val).lower()
                tokens = word_tokenize(lowercased_text)
                tokenized_docs.append(tokens)
                total_tokens += len(tokens)

        # Store complete token lists in dataframe and session artifacts
        session.dataframe["unigrams"] = tokenized_docs
        session.artifacts["unigrams"] = tokenized_docs

        # Build concise preview (first 3 documents, max 20 tokens each)
        preview = {}
        for i, tokens in enumerate(tokenized_docs[:3]):
            if len(tokens) > 20:
                preview[f"doc_{i}"] = tokens[:20] + ["..."]
            else:
                preview[f"doc_{i}"] = tokens

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "total_documents": len(tokenized_docs),
            "total_tokens": total_tokens,
            "token_preview": preview,
        }

        # Store complete tokenized results in results_store
        full_report = pd.DataFrame({"document_idx": range(len(tokenized_docs)), "unigrams": tokenized_docs})
        session.store_result("tokenize_tool", {"text_column": text_column}, summary, full_report=full_report)
        return summary

    @tool
    def sample_data_tool(
        n: int = 5,
        random_state: Optional[int] = None,
        replace_dataframe: bool = False,
    ) -> dict:
        """Subsamples rows from session.dataframe using pandas sample.

        Args:
            n: Number of rows to sample.
            random_state: Seed for reproducible random sampling.
            replace_dataframe: If True, replaces session.dataframe with the sampled subset.

        Returns:
            dict: Summary containing result_id, n_requested, sampled_rows,
                  sampled_indices, and replace_dataframe flag.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("session.dataframe is empty or None. Load a dataset first.")

        df = session.dataframe
        total_rows = len(df)

        if n < 1 or n > total_rows:
            raise ValueError(f"n must be between 1 and total_rows ({total_rows}).")

        sampled_df = df.sample(n=n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        n_preview = min(n, 10)
        sampled_records = sampled_df.head(n_preview).to_dict(orient="records")
        preview = _truncate_preview_records(sampled_records)

        if replace_dataframe:
            session.dataframe = sampled_df.reset_index(drop=True)

            # Invalidate stale DTM and document-aligned artifacts
            session.feature_matrix = None
            session.feature_names = None
            session.artifacts.pop("count_vectorizer", None)
            session.artifacts.pop("unigrams", None)

            # Resync labels
            for col in ["category_name", "category", "label"]:
                if col in session.dataframe.columns:
                    session.set_labels(session.dataframe[col].values)
                    break

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "n_requested": n,
            "sampled_rows": len(sampled_df),
            "sampled_indices": sampled_indices,
            "replace_dataframe": replace_dataframe,
            "preview": preview,
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "random_state": random_state, "replace_dataframe": replace_dataframe},
            summary,
        )
        return summary

    @tool
    def describe_data_tool(text_column: str = "text", category_column: str = "category_name") -> dict:
        """Computes summary statistics on text character length and plots length by category.

        Calculates text length (character count) for each document, computes standard
        .describe() summary statistics overall and broken down per category, and builds
        a box plot set on session.pending_figure.

        Args:
            text_column: Name of text column to analyze (default 'text').
            category_column: Name of category column for per-category stats and grouping (default 'category_name').

        Returns:
            dict: Summary containing result_id, overall descriptive statistics,
                  and per-category descriptive statistics.
        """
        if session.dataframe is None or session.dataframe.empty:
            raise ValueError("session.dataframe is empty or None. Load a dataset first.")

        if text_column not in session.dataframe.columns:
            raise ValueError(f"Column '{text_column}' not found in session.dataframe.")

        df = session.dataframe.copy()
        # Compute character count text_length per document
        df["text_length"] = df[text_column].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        session.dataframe["text_length"] = df["text_length"]

        # Overall summary statistics
        overall_stats = df["text_length"].describe().to_dict()

        # Per-category summary statistics
        by_category_stats = {}
        if category_column in df.columns:
            for cat, group in df.groupby(category_column):
                by_category_stats[str(cat)] = group["text_length"].describe().to_dict()

        # Build box plot of text_length by category
        fig, ax = plt.subplots(figsize=(8, 5))
        if category_column in df.columns:
            sns.boxplot(data=df, x=category_column, y="text_length", ax=ax)
            ax.set_title(f"Text Length Distribution by {category_column}")
            ax.set_xlabel(category_column)
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall": overall_stats,
            "by_category": by_category_stats,
        }

        session.store_result(
            "describe_data_tool",
            {"text_column": text_column, "category_column": category_column},
            summary,
        )
        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        list_files_tool,
        inspect_data_tool,
        tokenize_tool,
        sample_data_tool,
        describe_data_tool,
    ]
