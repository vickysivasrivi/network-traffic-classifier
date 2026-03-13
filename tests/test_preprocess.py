"""Unit tests for ``src.preprocess`` using synthetic DataFrames."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import (
    apply_smote,
    clean_dataframe,
    encode_labels,
    scale_features,
    split_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """200-row DataFrame with 4 numeric features and an imbalanced Label column."""
    rng = np.random.default_rng(seed=42)
    n: int = 200
    return pd.DataFrame(
        {
            "flow_duration": rng.uniform(0, 1_000, n),
            "tot_fwd_pkts": rng.integers(1, 100, n),
            "flow_bytes_per_sec": rng.uniform(0, 1e6, n),
            "pkt_len_mean": rng.uniform(50, 1_500, n),
            "Label": ["BENIGN"] * 150 + ["DDoS"] * 30 + ["PortScan"] * 20,
        }
    )


@pytest.fixture()
def dirty_df() -> pd.DataFrame:
    """4-row DataFrame with inf, high-NaN, single-NaN, constant, and spaced-label columns."""
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, np.inf, 4.0],
            "feature_b": [np.nan, np.nan, np.nan, 4.0],
            "feature_c": [5.0, np.nan, 7.0, 8.0],
            "constant_col": [0.0, 0.0, 0.0, 0.0],
            " Label": [" BENIGN", "DDoS", "BENIGN", "DDoS"],
        }
    )


# ---------------------------------------------------------------------------
# TestCleanDataframe
# ---------------------------------------------------------------------------


class TestCleanDataframe:
    """Tests for :func:`src.preprocess.clean_dataframe`."""

    def test_replaces_infinity_values(self, dirty_df: pd.DataFrame) -> None:
        """No ``inf`` values should remain after cleaning."""
        cleaned: pd.DataFrame = clean_dataframe(dirty_df.copy(), nan_threshold=0.5)
        numeric_cols: pd.Index = cleaned.select_dtypes(include=[np.number]).columns
        assert not np.isinf(cleaned[numeric_cols].values).any()

    def test_drops_high_nan_columns(self, dirty_df: pd.DataFrame) -> None:
        """Column with 75 % NaN (``feature_b``) must be dropped."""
        cleaned: pd.DataFrame = clean_dataframe(dirty_df.copy(), nan_threshold=0.5)
        assert "feature_b" not in cleaned.columns

    def test_keeps_low_nan_columns(self, dirty_df: pd.DataFrame) -> None:
        """Column with 25 % NaN (``feature_c``) must be retained."""
        cleaned: pd.DataFrame = clean_dataframe(dirty_df.copy(), nan_threshold=0.5)
        assert "feature_c" in cleaned.columns

    def test_drops_constant_columns(self, dirty_df: pd.DataFrame) -> None:
        """Zero-variance column (``constant_col``) must be dropped."""
        cleaned: pd.DataFrame = clean_dataframe(dirty_df.copy(), nan_threshold=0.5)
        assert "constant_col" not in cleaned.columns

    def test_normalises_label_column_name(self, dirty_df: pd.DataFrame) -> None:
        """Label column name must be normalised to ``"Label"``."""
        cleaned: pd.DataFrame = clean_dataframe(dirty_df.copy(), nan_threshold=0.5)
        assert "Label" in cleaned.columns

    def test_strips_label_values(self, dirty_df: pd.DataFrame) -> None:
        """Label string values must have no leading or trailing whitespace."""
        cleaned: pd.DataFrame = clean_dataframe(dirty_df.copy(), nan_threshold=0.5)
        assert all(v == v.strip() for v in cleaned["Label"])

    def test_no_nan_in_numeric_columns_after_cleaning(
        self, dirty_df: pd.DataFrame
    ) -> None:
        """No NaN must remain in numeric columns after cleaning."""
        cleaned: pd.DataFrame = clean_dataframe(dirty_df.copy(), nan_threshold=0.5)
        numeric_cols: pd.Index = cleaned.select_dtypes(include=[np.number]).columns
        assert cleaned[numeric_cols].isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# TestEncodeLabels
# ---------------------------------------------------------------------------


class TestEncodeLabels:
    """Tests for :func:`src.preprocess.encode_labels`."""

    def test_label_column_contains_integers(self, sample_df: pd.DataFrame) -> None:
        """``"Label"`` column must be integer dtype after encoding."""
        df_encoded, _ = encode_labels(sample_df.copy())
        assert pd.api.types.is_integer_dtype(df_encoded["Label"])

    def test_encoder_registers_all_classes(self, sample_df: pd.DataFrame) -> None:
        """Encoder must contain all three label classes."""
        _, le = encode_labels(sample_df.copy())
        assert set(le.classes_) == {"BENIGN", "DDoS", "PortScan"}

    def test_inverse_transform_recovers_original_labels(
        self, sample_df: pd.DataFrame
    ) -> None:
        """``inverse_transform`` on encoded labels must return the original strings."""
        df_encoded, le = encode_labels(sample_df.copy())
        recovered: np.ndarray = le.inverse_transform(df_encoded["Label"].values)
        assert list(recovered) == list(sample_df["Label"])


# ---------------------------------------------------------------------------
# TestSplitData
# ---------------------------------------------------------------------------


class TestSplitData:
    """Tests for :func:`src.preprocess.split_data`."""

    def test_split_sizes_are_proportional(self, sample_df: pd.DataFrame) -> None:
        """Train + test row count must equal the original; test set ~20 %."""
        df_encoded, _ = encode_labels(sample_df.copy())
        X_train, X_test, _, _ = split_data(df_encoded, test_size=0.2)
        total: int = len(X_train) + len(X_test)
        assert total == len(df_encoded)
        assert abs(len(X_test) - int(0.2 * total)) <= 5

    def test_label_column_absent_from_feature_matrices(
        self, sample_df: pd.DataFrame
    ) -> None:
        """``"Label"`` must not appear in ``X_train`` or ``X_test``."""
        df_encoded, _ = encode_labels(sample_df.copy())
        X_train, X_test, _, _ = split_data(df_encoded, test_size=0.2)
        assert "Label" not in X_train.columns
        assert "Label" not in X_test.columns

    def test_train_and_test_indices_are_disjoint(
        self, sample_df: pd.DataFrame
    ) -> None:
        """No row index must appear in both splits."""
        df_encoded, _ = encode_labels(sample_df.copy())
        X_train, X_test, _, _ = split_data(df_encoded, test_size=0.2)
        overlap: set[int] = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0


# ---------------------------------------------------------------------------
# TestApplySmote
# ---------------------------------------------------------------------------


class TestApplySmote:
    """Tests for :func:`src.preprocess.apply_smote`."""

    def test_minority_class_count_increases_after_smote(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Minority class count must be higher after SMOTE."""
        df_encoded, _ = encode_labels(sample_df.copy())
        X_train, _, y_train, _ = split_data(df_encoded, test_size=0.2, random_state=42)
        counts_before: pd.Series = pd.Series(y_train).value_counts()

        _, y_resampled = apply_smote(X_train, y_train)
        counts_after: pd.Series = pd.Series(y_resampled).value_counts()

        minority_class: int = int(counts_before.idxmin())
        assert counts_after[minority_class] > counts_before[minority_class]

    def test_total_sample_count_does_not_decrease(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Total sample count must not decrease after SMOTE."""
        df_encoded, _ = encode_labels(sample_df.copy())
        X_train, _, y_train, _ = split_data(df_encoded, test_size=0.2)
        _, y_resampled = apply_smote(X_train, y_train)
        assert len(y_resampled) >= len(y_train)

    def test_feature_and_label_arrays_have_matching_lengths(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Resampled ``X`` and ``y`` must have the same number of rows."""
        df_encoded, _ = encode_labels(sample_df.copy())
        X_train, _, y_train, _ = split_data(df_encoded, test_size=0.2)
        X_resampled, y_resampled = apply_smote(X_train, y_train)
        assert len(X_resampled) == len(y_resampled)


# ---------------------------------------------------------------------------
# TestScaleFeatures
# ---------------------------------------------------------------------------


class TestScaleFeatures:
    """Tests for :func:`src.preprocess.scale_features`."""

    def test_scaled_training_data_has_near_zero_mean(
        self, sample_df: pd.DataFrame
    ) -> None:
        """All feature column means in the scaled training matrix must be near zero."""
        df_encoded, _ = encode_labels(sample_df.copy())
        X_train, X_test, _, _ = split_data(df_encoded, test_size=0.2)
        X_train_scaled, _, _ = scale_features(X_train.values, X_test)
        column_means: np.ndarray = np.abs(X_train_scaled.mean(axis=0))
        assert np.all(column_means < 0.01), f"Column means not near zero: {column_means}"

    def test_scaler_exposes_training_statistics(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Returned scaler must have non-null ``mean_`` and ``scale_`` attributes."""
        df_encoded, _ = encode_labels(sample_df.copy())
        X_train, X_test, _, _ = split_data(df_encoded, test_size=0.2)
        _, _, scaler = scale_features(X_train.values, X_test)
        assert scaler.mean_ is not None
        assert scaler.scale_ is not None

    def test_test_output_preserves_shape(self, sample_df: pd.DataFrame) -> None:
        """Scaled test matrix must have the same shape as the input."""
        df_encoded, _ = encode_labels(sample_df.copy())
        X_train, X_test, _, _ = split_data(df_encoded, test_size=0.2)
        _, X_test_scaled, _ = scale_features(X_train.values, X_test)
        assert X_test_scaled.shape == X_test.shape
