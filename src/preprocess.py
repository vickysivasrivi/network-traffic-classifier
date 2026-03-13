"""Data cleaning, feature engineering, and train/test split for CICIDS 2017."""

import logging
import os
import gc
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log: logging.Logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict[str, Any]:
    """Load pipeline parameters from a YAML file.

    Args:
        params_path: Path to the YAML config file.

    Returns:
        Nested dictionary of pipeline parameters.

    Raises:
        FileNotFoundError: If ``params_path`` does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    with open(params_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_raw_data(raw_dir: str) -> pd.DataFrame:
    """Load and concatenate all CICIDS CSV files from a directory.

    Args:
        raw_dir: Directory containing raw ``.csv`` files.

    Returns:
        Single concatenated ``DataFrame`` with stripped column names.

    Raises:
        FileNotFoundError: If no ``.csv`` files are found in ``raw_dir``.
    """
    csv_files: list[str] = sorted(
        f for f in os.listdir(raw_dir) if f.endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{raw_dir}'.")

    frames: list[pd.DataFrame] = []
    for fname in csv_files:
        fpath: str = os.path.join(raw_dir, fname)
        log.info("Loading '%s' ...", fname)
        df: pd.DataFrame = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)

    combined: pd.DataFrame = pd.concat(frames, ignore_index=True)
    log.info(
        "Total rows loaded: %s  |  Columns: %d",
        f"{len(combined):,}",
        len(combined.columns),
    )
    return combined


def clean_dataframe(
    df: pd.DataFrame,
    nan_threshold: float = 0.5,
) -> pd.DataFrame:
    """Clean a raw CICIDS DataFrame.

    Steps: replace ``inf`` with ``NaN``, drop high-NaN and constant columns,
    fill remaining ``NaN`` with column median, normalise the ``"Label"`` column.

    Args:
        df: Raw DataFrame loaded from CICIDS CSV files.
        nan_threshold: Drop columns where NaN fraction exceeds this value.

    Returns:
        Cleaned ``DataFrame`` with a normalised ``"Label"`` column.

    Raises:
        ValueError: If no ``"Label"`` column is found.
    """
    log.info("Cleaning dataframe ...")

    df = df.replace([np.inf, -np.inf], np.nan)

    nan_fractions: pd.Series = df.isnull().mean()
    high_nan_cols: list[str] = (
        nan_fractions[nan_fractions > nan_threshold].index.tolist()
    )
    if high_nan_cols:
        log.info("Dropping %d high-NaN columns: %s", len(high_nan_cols), high_nan_cols)
        df = df.drop(columns=high_nan_cols)

    constant_cols: list[str] = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        log.info("Dropping %d constant columns.", len(constant_cols))
        df = df.drop(columns=constant_cols)

    numeric_cols: pd.Index = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    label_col: str | None = next(
        (c for c in df.columns if c.strip().lower() == "label"), None
    )
    if label_col is None:
        raise ValueError(
            "Could not find a 'Label' column. "
            f"Available columns: {df.columns.tolist()}"
        )
    if label_col != "Label":
        df = df.rename(columns={label_col: "Label"})

    df["Label"] = df["Label"].str.strip()

    # Optimize memory usage by downcasting
    log.info("Optimizing memory usage by downcasting numeric types ...")
    float_cols = df.select_dtypes(include=['float64']).columns
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    gc.collect()

    log.info("After cleaning: %s rows, %d columns.", f"{len(df):,}", len(df.columns))
    log.info("Class distribution:\n%s", df["Label"].value_counts().to_string())
    return df


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Encode the ``"Label"`` column from strings to integers.

    Args:
        df: Cleaned DataFrame with a string ``"Label"`` column.

    Returns:
        Tuple of the modified ``DataFrame`` and the fitted ``LabelEncoder``.
    """
    le: LabelEncoder = LabelEncoder()
    df["Label"] = le.fit_transform(df["Label"])
    log.info("Encoded %d classes: %s", len(le.classes_), list(le.classes_))
    return df, le


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train/test split.

    Args:
        df: DataFrame with an integer-encoded ``"Label"`` column.
        test_size: Fraction reserved for the test set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test)``.
    """
    X: pd.DataFrame = df.drop(columns=["Label"])
    y: pd.Series = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    log.info(
        "Split — Train: %s rows  |  Test: %s rows.",
        f"{len(X_train):,}",
        f"{len(X_test):,}",
    )
    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes on training data using SMOTE.

    Args:
        X_train: Training feature matrix.
        y_train: Integer-encoded training labels.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple ``(X_resampled, y_resampled)`` as ``np.ndarray`` arrays.
    """
    log.info("Applying Undersampling + SMOTE Pipeline to training data ...")
    log.info(
        "Before SMOTE — class distribution:\n%s",
        pd.Series(y_train).value_counts().to_string(),
    )

    # Strategy:
    # 1. Under-sample any class with > 100,000 samples down to 100,000
    # 2. Over-sample any class with < 100,000 samples up to 100,000
    # This keeps total memory strictly bounded.
    counts = pd.Series(y_train).value_counts()
    under_strategy = {c: min(count, 100000) for c, count in counts.items()}
    over_strategy = {c: 100000 for c in counts.keys()}

    pipeline = Pipeline([
        ('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)),
        ('over', SMOTE(sampling_strategy=over_strategy, random_state=random_state))
    ])
    
    X_resampled: np.ndarray
    y_resampled: np.ndarray
    X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)

    log.info(
        "After SMOTE — class distribution:\n%s",
        pd.Series(y_resampled).value_counts().to_string(),
    )
    return X_resampled, y_resampled


def scale_features(
    X_train: np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardise features to zero mean and unit variance.

    Scaler is fitted on training data only to prevent data leakage.

    Args:
        X_train: Training feature matrix (post-SMOTE).
        X_test: Test feature matrix.

    Returns:
        Tuple ``(X_train_scaled, X_test_scaled, scaler)``.
    """
    X_test_array: np.ndarray = (
        X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    )

    scaler: StandardScaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    X_test_scaled: np.ndarray = scaler.transform(X_test_array)
    return X_train_scaled, X_test_scaled, scaler


def save_processed(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: pd.Series,
    feature_names: list[str],
    processed_dir: str,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
) -> None:
    """Save all processed arrays and fitted transformers to ``processed_dir``.

    Args:
        X_train: Scaled training features.
        X_test: Scaled test features.
        y_train: SMOTE-balanced training labels.
        y_test: Test labels.
        feature_names: Ordered list of feature column names.
        processed_dir: Output directory (created if missing).
        scaler: Fitted ``StandardScaler``.
        label_encoder: Fitted ``LabelEncoder``.
    """
    os.makedirs(processed_dir, exist_ok=True)

    artifacts: dict[str, Any] = {
        "X_train.joblib": X_train,
        "X_test.joblib": X_test,
        "y_train.joblib": y_train,
        "y_test.joblib": y_test,
        "scaler.joblib": scaler,
        "label_encoder.joblib": label_encoder,
    }
    for filename, obj in artifacts.items():
        joblib.dump(obj, os.path.join(processed_dir, filename))

    feature_names_path: str = os.path.join(processed_dir, "feature_names.txt")
    with open(feature_names_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(feature_names))

    log.info("Saved %d artifacts to '%s/'.", len(artifacts) + 1, processed_dir)


def run_preprocessing(params_path: str = "params.yaml") -> None:
    """Run the full preprocessing pipeline and save all artifacts.

    Invoked by DVC (``dvc repro preprocess``) or directly via
    ``python src/preprocess.py``.

    Args:
        params_path: Path to the YAML parameter file.
    """
    params: dict[str, Any] = load_params(params_path)

    raw_dir: str = params["data"]["raw_dir"]
    processed_dir: str = params["data"]["processed_dir"]
    test_size: float = params["data"]["test_size"]
    random_state: int = params["data"]["random_state"]
    nan_threshold: float = params["preprocessing"]["nan_threshold"]
    smote_seed: int = params["preprocessing"]["smote_random_state"]

    df: pd.DataFrame = load_raw_data(raw_dir)
    df = clean_dataframe(df, nan_threshold)
    df, label_encoder = encode_labels(df)

    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)
    feature_names: list[str] = X_train.columns.tolist()

    del df
    gc.collect()

    X_train_smote, y_train_smote = apply_smote(X_train, y_train, smote_seed)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_smote, X_test)

    save_processed(
        X_train_scaled,
        X_test_scaled,
        y_train_smote,
        y_test,
        feature_names,
        processed_dir,
        scaler,
        label_encoder,
    )
    log.info("Preprocessing complete.")


if __name__ == "__main__":
    run_preprocessing()
