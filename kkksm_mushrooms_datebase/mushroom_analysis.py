from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Binarizer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

RANDOM_STATE = 42
TARGET_COLUMN = "class"
COLUMNS = [
    "class",
    "cap_shape",
    "cap_surface",
    "cap_color",
    "bruises",
    "odor",
    "gill_attachment",
    "gill_spacing",
    "gill_size",
    "gill_color",
    "stalk_shape",
    "stalk_root",
    "stalk_surface_above_ring",
    "stalk_surface_below_ring",
    "stalk_color_above_ring",
    "stalk_color_below_ring",
    "veil_type",
    "veil_color",
    "ring_number",
    "ring_type",
    "spore_print_color",
    "population",
    "habitat",
]
FEATURE_COLUMNS = [column for column in COLUMNS if column != TARGET_COLUMN]
FEATURES_FOR_HISTOGRAMS = [
    "odor",
    "gill_color",
    "cap_color",
    "habitat",
    "population",
    "bruises",
]
DATASET_CONFIGS = {
    "mushroom": {
        "target_column": TARGET_COLUMN,
        "positive_label": "poisonous",
        "feature_histograms": FEATURES_FOR_HISTOGRAMS,
    },
    "adult_income": {
        "target_column": TARGET_COLUMN,
        "positive_label": ">50K",
        "feature_histograms": [
            "age",
            "education_num",
            "hours_per_week",
            "workclass",
            "occupation",
            "marital_status",
        ],
    },
    "breast_cancer": {
        "target_column": TARGET_COLUMN,
        "positive_label": "malignant",
        "feature_histograms": [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean concavity",
            "worst concave points",
        ],
    },
}
MUSHROOM_VALUE_LABELS = {
    "class": {"e": "edible", "p": "poisonous"},
    "cap_shape": {
        "b": "bell",
        "c": "conical",
        "x": "convex",
        "f": "flat",
        "k": "knobbed",
        "s": "sunken",
    },
    "cap_surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
    "cap_color": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "r": "green",
        "p": "pink",
        "u": "purple",
        "e": "red",
        "w": "white",
        "y": "yellow",
    },
    "bruises": {"t": "bruises", "f": "no bruises"},
    "odor": {
        "a": "almond",
        "l": "anise",
        "c": "creosote",
        "y": "fishy",
        "f": "foul",
        "m": "musty",
        "n": "none",
        "p": "pungent",
        "s": "spicy",
    },
    "gill_attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
    "gill_spacing": {"c": "close", "w": "crowded", "d": "distant"},
    "gill_size": {"b": "broad", "n": "narrow"},
    "gill_color": {
        "k": "black",
        "n": "brown",
        "b": "buff",
        "h": "chocolate",
        "g": "gray",
        "r": "green",
        "o": "orange",
        "p": "pink",
        "u": "purple",
        "e": "red",
        "w": "white",
        "y": "yellow",
    },
    "stalk_shape": {"e": "enlarging", "t": "tapering"},
    "stalk_root": {"b": "bulbous", "c": "club", "u": "cup", "e": "equal", "z": "rhizomorphs", "r": "rooted"},
    "stalk_surface_above_ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk_surface_below_ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk_color_above_ring": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "o": "orange",
        "p": "pink",
        "e": "red",
        "w": "white",
        "y": "yellow",
    },
    "stalk_color_below_ring": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "o": "orange",
        "p": "pink",
        "e": "red",
        "w": "white",
        "y": "yellow",
    },
    "veil_type": {"p": "partial", "u": "universal"},
    "veil_color": {"n": "brown", "o": "orange", "w": "white", "y": "yellow"},
    "ring_number": {"n": "none", "o": "one", "t": "two"},
    "ring_type": {"c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large", "n": "none", "p": "pendant", "s": "sheathing", "z": "zone"},
    "spore_print_color": {
        "k": "black",
        "n": "brown",
        "b": "buff",
        "h": "chocolate",
        "r": "green",
        "o": "orange",
        "u": "purple",
        "w": "white",
        "y": "yellow",
    },
    "population": {"a": "abundant", "c": "clustered", "n": "numerous", "s": "scattered", "v": "several", "y": "solitary"},
    "habitat": {"g": "grasses", "l": "leaves", "m": "meadows", "p": "paths", "u": "urban", "w": "waste", "d": "woods"},
}


def get_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "agaricus-lepiota.data"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(get_data_path(), header=None, names=COLUMNS)
    return df.replace("?", np.nan)


def load_breast_cancer_data() -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    # Target mapping keeps a readable class label in the same "class" column used elsewhere.
    df[TARGET_COLUMN] = df["target"].map({0: "malignant", 1: "benign"})
    df = df.drop(columns=["target"])
    return df


def load_adult_income_data() -> pd.DataFrame:
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        TARGET_COLUMN,
    ]
    data_path = Path(__file__).resolve().parents[1] / "adult.data"
    df = pd.read_csv(data_path, header=None, names=column_names, skipinitialspace=True, na_values="?")
    return df


def load_dataset(dataset_name: str = "mushroom") -> pd.DataFrame:
    if dataset_name == "mushroom":
        return load_data()
    if dataset_name == "adult_income":
        return load_adult_income_data()
    if dataset_name == "breast_cancer":
        return load_breast_cancer_data()
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def decode_mushroom_value(column: str, value: object) -> str:
    if pd.isna(value):
        return "missing"
    mapping = MUSHROOM_VALUE_LABELS.get(column, {})
    return mapping.get(value, str(value))


def get_human_readable_class_labels(class_values: list[str] | pd.Index) -> list[str]:
    return [decode_mushroom_value(TARGET_COLUMN, value) for value in class_values]


def humanize_encoded_feature(feature_name: str, feature_columns: list[str]) -> str:
    for column in sorted(feature_columns, key=len, reverse=True):
        prefix = f"{column}_"
        if feature_name.startswith(prefix):
            raw_value = feature_name[len(prefix) :]
            return f"{column} = {decode_mushroom_value(column, raw_value)}"
    return feature_name


def class_balance_table(target: pd.Series) -> pd.DataFrame:
    counts = target.value_counts().sort_index()
    percentages = counts.div(len(target)).mul(100).round(2)
    labels = pd.Series(counts.index, index=counts.index).replace({"e": "edible", "p": "poisonous"}).values
    return pd.DataFrame(
        {
            "class_code": counts.index,
            "class_name": labels,
            "count": counts.values,
            "percentage": percentages.values,
        }
    )


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_COLUMN],
    )
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(
        drop=True
    )


def missing_values_table(*datasets: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, dataset in datasets:
        missing_counts = dataset.isna().sum()
        for column, missing_count in missing_counts.items():
            if missing_count:
                rows.append(
                    {
                        "split": split_name,
                        "column": column,
                        "missing_count": int(missing_count),
                    }
                )
    return pd.DataFrame(rows)


def preprocess_after_split(
    train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict[str, object]:
    feature_columns = [column for column in train_df.columns if column != TARGET_COLUMN]
    categorical_columns = train_df[feature_columns].select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    categorical_imputer = SimpleImputer(strategy="most_frequent") if categorical_columns else None
    numeric_imputer = SimpleImputer(strategy="median") if numeric_columns else None
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False) if categorical_columns else None
    scaler = StandardScaler() if numeric_columns else None
    binarizer = Binarizer(threshold=0.0)
    label_encoder = LabelEncoder()

    numeric_frames: dict[str, pd.DataFrame] = {}
    categorical_frames: dict[str, pd.DataFrame] = {}

    if numeric_columns:
        X_train_numeric = pd.DataFrame(
            numeric_imputer.fit_transform(train_df[numeric_columns]), columns=numeric_columns
        )
        X_validation_numeric = pd.DataFrame(
            numeric_imputer.transform(validation_df[numeric_columns]), columns=numeric_columns
        )
        X_test_numeric = pd.DataFrame(numeric_imputer.transform(test_df[numeric_columns]), columns=numeric_columns)

        numeric_frames["train"] = pd.DataFrame(scaler.fit_transform(X_train_numeric), columns=numeric_columns)
        numeric_frames["validation"] = pd.DataFrame(
            scaler.transform(X_validation_numeric), columns=numeric_columns
        )
        numeric_frames["test"] = pd.DataFrame(scaler.transform(X_test_numeric), columns=numeric_columns)

    if categorical_columns:
        X_train_categorical = pd.DataFrame(
            categorical_imputer.fit_transform(train_df[categorical_columns]), columns=categorical_columns
        )
        X_validation_categorical = pd.DataFrame(
            categorical_imputer.transform(validation_df[categorical_columns]), columns=categorical_columns
        )
        X_test_categorical = pd.DataFrame(
            categorical_imputer.transform(test_df[categorical_columns]), columns=categorical_columns
        )

        categorical_frames["train"] = pd.DataFrame(
            encoder.fit_transform(X_train_categorical),
            columns=encoder.get_feature_names_out(categorical_columns),
        )
        categorical_frames["validation"] = pd.DataFrame(
            encoder.transform(X_validation_categorical),
            columns=encoder.get_feature_names_out(categorical_columns),
        )
        categorical_frames["test"] = pd.DataFrame(
            encoder.transform(X_test_categorical),
            columns=encoder.get_feature_names_out(categorical_columns),
        )

    y_train = label_encoder.fit_transform(train_df[TARGET_COLUMN])
    y_validation = label_encoder.transform(validation_df[TARGET_COLUMN])
    y_test = label_encoder.transform(test_df[TARGET_COLUMN])

    def _combine_frames(split_name: str) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        if numeric_columns:
            parts.append(numeric_frames[split_name].reset_index(drop=True))
        if categorical_columns:
            parts.append(categorical_frames[split_name].reset_index(drop=True))
        if not parts:
            return pd.DataFrame(index=range(len(train_df if split_name == "train" else validation_df if split_name == "validation" else test_df)))
        return pd.concat(parts, axis=1)

    X_train_model = _combine_frames("train")
    X_validation_model = _combine_frames("validation")
    X_test_model = _combine_frames("test")

    X_train_binary = pd.DataFrame(binarizer.fit_transform(X_train_model), columns=X_train_model.columns)
    X_validation_binary = pd.DataFrame(
        binarizer.transform(X_validation_model), columns=X_validation_model.columns
    )
    X_test_binary = pd.DataFrame(binarizer.transform(X_test_model), columns=X_test_model.columns)

    X_train_raw = train_df[feature_columns].copy()
    X_validation_raw = validation_df[feature_columns].copy()
    X_test_raw = test_df[feature_columns].copy()

    return {
        "categorical_imputer": categorical_imputer,
        "numeric_imputer": numeric_imputer,
        "encoder": encoder,
        "scaler": scaler,
        "binarizer": binarizer,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "X_train_raw": X_train_raw,
        "X_validation_raw": X_validation_raw,
        "X_test_raw": X_test_raw,
        "X_train_model": X_train_model,
        "X_validation_model": X_validation_model,
        "X_test_model": X_test_model,
        "X_train_binary": X_train_binary,
        "X_validation_binary": X_validation_binary,
        "X_test_binary": X_test_binary,
        "X_train_encoded": X_train_model,
        "X_validation_encoded": X_validation_model,
        "X_test_encoded": X_test_model,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
    }


def oversample_minority_class(
    X: pd.DataFrame, y: pd.Series | list[int]
) -> tuple[pd.DataFrame, pd.Series]:
    train_encoded = X.copy()
    train_encoded["target"] = pd.Series(y).reset_index(drop=True)

    class_counts = train_encoded["target"].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    majority_df = train_encoded[train_encoded["target"] == majority_class]
    minority_df = train_encoded[train_encoded["target"] == minority_class]

    minority_oversampled = resample(
        minority_df,
        replace=True,
        n_samples=len(majority_df),
        random_state=RANDOM_STATE,
    )

    balanced_df = pd.concat([majority_df, minority_oversampled], ignore_index=True).sample(
        frac=1.0, random_state=RANDOM_STATE
    )

    return balanced_df.drop(columns="target"), balanced_df["target"]


def undersample_majority_class(
    X: pd.DataFrame, y: pd.Series | list[int]
) -> tuple[pd.DataFrame, pd.Series]:
    train_encoded = X.copy()
    train_encoded["target"] = pd.Series(y).reset_index(drop=True)

    class_counts = train_encoded["target"].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    majority_df = train_encoded[train_encoded["target"] == majority_class]
    minority_df = train_encoded[train_encoded["target"] == minority_class]

    majority_undersampled = resample(
        majority_df,
        replace=False,
        n_samples=len(minority_df),
        random_state=RANDOM_STATE,
    )

    balanced_df = pd.concat([majority_undersampled, minority_df], ignore_index=True).sample(
        frac=1.0, random_state=RANDOM_STATE
    )

    return balanced_df.drop(columns="target"), balanced_df["target"]


def build_model_candidates() -> dict[str, dict[str, object]]:
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                max_depth=12, min_samples_leaf=4, class_weight="balanced", random_state=RANDOM_STATE
            )
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        },
        "KNN": {
            "model": KNeighborsClassifier(n_neighbors=11),
            "balancing": "oversample",
        },
        "Bernoulli Naive Bayes": {
            "model": BernoulliNB(),
            "balancing": "oversample",
        },
    }


def evaluate_models(preprocessed: dict[str, object]) -> pd.DataFrame:
    X_train = preprocessed["X_train_model"]
    X_validation = preprocessed["X_validation_model"]
    X_test = preprocessed["X_test_model"]
    X_train_binary = preprocessed["X_train_binary"]
    X_validation_binary = preprocessed["X_validation_binary"]
    X_test_binary = preprocessed["X_test_binary"]
    y_train = pd.Series(preprocessed["y_train"])
    y_validation = pd.Series(preprocessed["y_validation"])
    y_test = pd.Series(preprocessed["y_test"])

    results: list[dict[str, object]] = []
    for model_name, config in build_model_candidates().items():
        X_fit = X_train
        y_fit = y_train
        X_validation_eval = X_validation
        X_test_eval = X_test
        balancing = config.get("balancing", "class_weight")

        if model_name == "Bernoulli Naive Bayes":
            X_fit = X_train_binary
            X_validation_eval = X_validation_binary
            X_test_eval = X_test_binary

        if balancing == "oversample":
            X_fit, y_fit = oversample_minority_class(X_train, y_train)
            if model_name == "Bernoulli Naive Bayes":
                X_fit, y_fit = oversample_minority_class(X_train_binary, y_train)

        model = config["model"]
        model.fit(X_fit, y_fit)

        validation_predictions = model.predict(X_validation_eval)
        test_predictions = model.predict(X_test_eval)

        results.append(
            {
                "model": model_name,
                "balancing_strategy": balancing,
                "validation_accuracy": round(accuracy_score(y_validation, validation_predictions), 4),
                "validation_precision": round(
                    precision_score(y_validation, validation_predictions, zero_division=0), 4
                ),
                "validation_recall": round(recall_score(y_validation, validation_predictions, zero_division=0), 4),
                "validation_f1": round(f1_score(y_validation, validation_predictions, zero_division=0), 4),
                "test_accuracy": round(accuracy_score(y_test, test_predictions), 4),
                "test_precision": round(precision_score(y_test, test_predictions, zero_division=0), 4),
                "test_recall": round(recall_score(y_test, test_predictions, zero_division=0), 4),
                "test_f1": round(f1_score(y_test, test_predictions, zero_division=0), 4),
                "fitted_model": model,
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["validation_f1", "validation_accuracy"], ascending=False
    )
    return results_df.reset_index(drop=True)


def fit_best_model(preprocessed: dict[str, object], results_df: pd.DataFrame) -> tuple[str, object]:
    best_row = results_df.iloc[0]
    model_name = best_row["model"]
    config = build_model_candidates()[model_name]

    X_fit = preprocessed["X_train_model"]
    y_fit = pd.Series(preprocessed["y_train"])

    if model_name == "Bernoulli Naive Bayes":
        X_fit = preprocessed["X_train_binary"]

    if config.get("balancing") == "oversample":
        X_fit, y_fit = oversample_minority_class(X_fit, y_fit)

    model = config["model"]
    model.fit(X_fit, y_fit)
    return model_name, model


def plot_class_distribution(target: pd.Series, dataset_name: str = "mushroom") -> None:
    table = class_balance_table(target)
    plt.figure(figsize=(7, 4))
    plt.pie(
        table["count"],
        labels=table["class_name"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2a9d8f", "#e76f51"],
    )
    plt.title(f"Class distribution in the {dataset_name} dataset")
    plt.tight_layout()
    plt.show()


def plot_feature_histograms(
    dataset: pd.DataFrame, columns: list[str] | None = None, dataset_name: str = "mushroom"
) -> None:
    default_columns = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["mushroom"])["feature_histograms"]
    selected_columns = columns or default_columns
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for axis, column in zip(axes, selected_columns):
        if pd.api.types.is_numeric_dtype(dataset[column]):
            axis.hist(dataset[column].dropna(), bins=20, color="#2a9d8f", edgecolor="black")
        else:
            counts = dataset[column].value_counts().sort_values(ascending=False)
            labels = [
                decode_mushroom_value(column, value) if dataset_name == "mushroom" else str(value)
                for value in counts.index
            ]
            axis.bar(labels, counts.values, color="#2a9d8f")
            axis.tick_params(axis="x", rotation=45)
        axis.set_title(f"{column} distribution")
        axis.set_xlabel(column)
        axis.set_ylabel("Count")

    for axis in axes[len(selected_columns) :]:
        axis.remove()

    plt.tight_layout()
    plt.show()


def plot_top_feature_correlations(
    X_train_encoded: pd.DataFrame,
    y_train: list[int] | pd.Series,
    top_n: int = 12,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    target_series = pd.Series(y_train, name="target")
    target_correlations = X_train_encoded.corrwith(target_series).abs().sort_values(ascending=False).head(top_n)
    correlation_frame = X_train_encoded[target_correlations.index].copy()
    correlation_frame["target"] = target_series
    # Constant columns produce undefined correlation values and noisy runtime warnings.
    non_constant_columns = [column for column in correlation_frame.columns if correlation_frame[column].nunique() > 1]
    correlation_frame = correlation_frame[non_constant_columns]

    plt.figure(figsize=(12, 8))
    plt.imshow(correlation_frame.corr(), cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(correlation_frame.columns)), correlation_frame.columns, rotation=90)
    plt.yticks(range(len(correlation_frame.columns)), correlation_frame.columns)
    plt.title("Correlation matrix for the most informative encoded features")
    plt.tight_layout()
    plt.show()

    output = pd.DataFrame(
        {"encoded_feature": target_correlations.index, "absolute_target_correlation": target_correlations.values}
    )
    if feature_columns:
        output["feature_description"] = output["encoded_feature"].apply(
            lambda value: humanize_encoded_feature(value, feature_columns)
        )
    return output


def plot_confusion_matrix_for_model(
    model: object, X_test_encoded: pd.DataFrame, y_test: list[int] | pd.Series, label_encoder: LabelEncoder
) -> pd.DataFrame:
    predictions = model.predict(X_test_encoded)
    raw_labels = label_encoder.classes_
    labels = get_human_readable_class_labels(raw_labels)
    matrix = confusion_matrix(y_test, predictions)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix_df, cmap="Blues", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(matrix_df.columns)), matrix_df.columns)
    plt.yticks(range(len(matrix_df.index)), matrix_df.index)
    for row_index in range(matrix_df.shape[0]):
        for column_index in range(matrix_df.shape[1]):
            plt.text(column_index, row_index, matrix_df.iloc[row_index, column_index], ha="center", va="center")
    plt.title("Confusion matrix on the test set")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()


def plot_model_comparison(results_df: pd.DataFrame, dataset_name: str) -> None:
    comparison = results_df.drop(columns=["fitted_model"], errors="ignore").copy()
    models = comparison["model"]
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(models))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for axis, split_name in zip(axes, ["validation", "test"]):
        for index, metric in enumerate(metrics):
            axis.bar(
                x + (index - 1.5) * width,
                comparison[f"{split_name}_{metric}"],
                width=width,
                label=metric,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(models, rotation=20, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Score")
        axis.set_title(f"{split_name.capitalize()} metrics")

    fig.suptitle(f"Comparison of model quality on the {dataset_name} dataset")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    plt.tight_layout()
    plt.show()


def run_analysis(dataset_name: str = "mushroom", print_summary: bool = True) -> dict[str, object]:
    df = load_dataset(dataset_name)
    train_df, validation_df, test_df = split_dataset(df)
    preprocessed = preprocess_after_split(train_df, validation_df, test_df)
    results_df = evaluate_models(preprocessed)
    best_model_name, best_model = fit_best_model(preprocessed, results_df)

    if print_summary:
        print("Dataset shape:", df.shape)
        print("\nClass balance:")
        print(class_balance_table(df[TARGET_COLUMN]).to_string(index=False))
        print("\nSplit sizes:")
        print(
            pd.DataFrame(
                [
                    {"split": "train", "rows": len(train_df)},
                    {"split": "validation", "rows": len(validation_df)},
                    {"split": "test", "rows": len(test_df)},
                ]
            ).to_string(index=False)
        )
        print("\nMissing values by split:")
        print(
            missing_values_table(
                ("train", train_df),
                ("validation", validation_df),
                ("test", test_df),
            ).to_string(index=False)
        )
        print("\nModel comparison:")
        printable_results = results_df.drop(columns="fitted_model")
        print(printable_results.to_string(index=False))
        print(f"\nBest model by validation F1: {best_model_name}")

    return {
        "dataset_name": dataset_name,
        "df": df,
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,
        "preprocessed": preprocessed,
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_model": best_model,
    }


def compare_datasets(dataset_names: list[str] | None = None) -> pd.DataFrame:
    datasets_to_compare = dataset_names or ["mushroom", "adult_income"]
    comparison_frames: list[pd.DataFrame] = []

    for dataset_name in datasets_to_compare:
        analysis = run_analysis(dataset_name=dataset_name, print_summary=False)
        dataset_results = analysis["results_df"].drop(columns="fitted_model").copy()
        dataset_results.insert(0, "dataset", dataset_name)
        comparison_frames.append(dataset_results)

    combined = pd.concat(comparison_frames, ignore_index=True)
    return combined.sort_values(by=["dataset", "validation_f1", "validation_accuracy"], ascending=[True, False, False])


if __name__ == "__main__":
    run_analysis(print_summary=True)
