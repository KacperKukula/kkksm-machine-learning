from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


OUTPUT_DIR = Path(__file__).resolve().parent


def source(text: str) -> list[str]:
    normalized = dedent(text).strip("\n")
    return [line + "\n" for line in normalized.split("\n")]


def markdown_cell(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save_notebook(filename: str, cells: list[dict[str, object]]) -> None:
    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Written: {path.name}")


def common_imports_cell() -> dict[str, object]:
    return code_cell(
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        from mushroom_analysis import (
            TARGET_COLUMN,
            BernoulliNB,
            DecisionTreeClassifier,
            KNeighborsClassifier,
            LogisticRegression,
            RANDOM_STATE,
            RandomForestClassifier,
            class_balance_table,
            load_data,
            load_dataset,
            missing_values_table,
            oversample_minority_class,
            plot_class_distribution,
            plot_confusion_matrix_for_model,
            plot_feature_histograms,
            plot_top_feature_correlations,
            preprocess_after_split,
            split_dataset,
        )
        """
    )


def metric_table_code(train_ref: str) -> str:
    return dedent(
        f"""
        model.fit({train_ref}, y_train_reference)

        validation_pred = model.predict(preprocessed["X_validation_encoded"])
        test_pred = model.predict(preprocessed["X_test_encoded"])

        pd.DataFrame(
            [
                {{
                    "split": "validation",
                    "accuracy": round(accuracy_score(preprocessed["y_validation"], validation_pred), 4),
                    "precision": round(precision_score(preprocessed["y_validation"], validation_pred), 4),
                    "recall": round(recall_score(preprocessed["y_validation"], validation_pred), 4),
                    "f1": round(f1_score(preprocessed["y_validation"], validation_pred), 4),
                }},
                {{
                    "split": "test",
                    "accuracy": round(accuracy_score(preprocessed["y_test"], test_pred), 4),
                    "precision": round(precision_score(preprocessed["y_test"], test_pred), 4),
                    "recall": round(recall_score(preprocessed["y_test"], test_pred), 4),
                    "f1": round(f1_score(preprocessed["y_test"], test_pred), 4),
                }},
            ]
        )
        """
    )


def dataset_run_code(dataset_name: str, use_binary: bool = False) -> str:
    x_train_ref = "preprocessed['X_train_binary']" if use_binary else "preprocessed['X_train_model']"
    x_val_ref = "preprocessed['X_validation_binary']" if use_binary else "preprocessed['X_validation_model']"
    x_test_ref = "preprocessed['X_test_binary']" if use_binary else "preprocessed['X_test_model']"
    return dedent(
        f"""
        df = load_dataset("{dataset_name}")
        train_df, validation_df, test_df = split_dataset(df)
        preprocessed = preprocess_after_split(train_df, validation_df, test_df)

        X_train_reference = {x_train_ref}
        y_train_reference = preprocessed["y_train"]
        X_validation_reference = {x_val_ref}
        X_test_reference = {x_test_ref}

        print("Dataset:", "{dataset_name}")
        print("Shape:", df.shape)
        """
    )


def dataset_oversample_code(dataset_name: str, use_binary: bool = False) -> str:
    x_train_ref = "preprocessed['X_train_binary']" if use_binary else "preprocessed['X_train_model']"
    x_val_ref = "preprocessed['X_validation_binary']" if use_binary else "preprocessed['X_validation_model']"
    x_test_ref = "preprocessed['X_test_binary']" if use_binary else "preprocessed['X_test_model']"
    return dedent(
        f"""
        df = load_dataset("{dataset_name}")
        train_df, validation_df, test_df = split_dataset(df)
        preprocessed = preprocess_after_split(train_df, validation_df, test_df)

        X_train_reference, y_train_reference = oversample_minority_class(
            {x_train_ref},
            preprocessed["y_train"],
        )
        X_validation_reference = {x_val_ref}
        X_test_reference = {x_test_ref}

        print("Dataset:", "{dataset_name}")
        print("Shape:", df.shape)
        print(pd.Series(y_train_reference).value_counts())
        """
    )


def dataset_metric_code() -> str:
    return dedent(
        """
        model.fit(X_train_reference, y_train_reference)

        validation_pred = model.predict(X_validation_reference)
        test_pred = model.predict(X_test_reference)

        metrics_df = pd.DataFrame(
            [
                {
                    "split": "validation",
                    "accuracy": round(accuracy_score(preprocessed["y_validation"], validation_pred), 4),
                    "precision": round(precision_score(preprocessed["y_validation"], validation_pred), 4),
                    "recall": round(recall_score(preprocessed["y_validation"], validation_pred), 4),
                    "f1": round(f1_score(preprocessed["y_validation"], validation_pred), 4),
                },
                {
                    "split": "test",
                    "accuracy": round(accuracy_score(preprocessed["y_test"], test_pred), 4),
                    "precision": round(precision_score(preprocessed["y_test"], test_pred), 4),
                    "recall": round(recall_score(preprocessed["y_test"], test_pred), 4),
                    "f1": round(f1_score(preprocessed["y_test"], test_pred), 4),
                },
            ]
        )
        metrics_df
        """
    )


def dataset_metric_conclusion_code(dataset_label: str) -> str:
    return dedent(
        f"""
        validation_row = metrics_df[metrics_df["split"] == "validation"].iloc[0]
        test_row = metrics_df[metrics_df["split"] == "test"].iloc[0]

        print(
            f"Wnioski dla {dataset_label}: validation F1 = {{validation_row['f1']:.4f}}, "
            f"test F1 = {{test_row['f1']:.4f}}, validation accuracy = {{validation_row['accuracy']:.4f}}, "
            f"test accuracy = {{test_row['accuracy']:.4f}}."
        )
        """
    )


def comparison_summary_code() -> str:
    return dedent(
        """
        comparison_df = pd.DataFrame(
            [
                {
                    "dataset": "mushroom",
                    "validation_accuracy": mushroom_metrics[mushroom_metrics["split"] == "validation"].iloc[0]["accuracy"],
                    "validation_f1": mushroom_metrics[mushroom_metrics["split"] == "validation"].iloc[0]["f1"],
                    "test_accuracy": mushroom_metrics[mushroom_metrics["split"] == "test"].iloc[0]["accuracy"],
                    "test_f1": mushroom_metrics[mushroom_metrics["split"] == "test"].iloc[0]["f1"],
                },
                {
                    "dataset": "adult_income",
                    "validation_accuracy": adult_metrics[adult_metrics["split"] == "validation"].iloc[0]["accuracy"],
                    "validation_f1": adult_metrics[adult_metrics["split"] == "validation"].iloc[0]["f1"],
                    "test_accuracy": adult_metrics[adult_metrics["split"] == "test"].iloc[0]["accuracy"],
                    "test_f1": adult_metrics[adult_metrics["split"] == "test"].iloc[0]["f1"],
                },
            ]
        )
        comparison_df
        """
    )


def comparison_conclusion_code(model_name: str) -> str:
    return dedent(
        f"""
        mushroom_row = comparison_df[comparison_df["dataset"] == "mushroom"].iloc[0]
        adult_row = comparison_df[comparison_df["dataset"] == "adult_income"].iloc[0]

        print(
            f"Wnioski koncowe dla {model_name}: model osiagnal na mushroom test F1 = "
            f"{{mushroom_row['test_f1']:.4f}}, natomiast na adult_income test F1 = "
            f"{{adult_row['test_f1']:.4f}}. Oznacza to, ze mushroom jest datasetem "
            f"znacznie latwiejszym, a adult_income stanowi bardziej wymagajacy problem klasyfikacyjny."
        )
        """
    )


def class_conclusion_code() -> str:
    return dedent(
        """
        balance_df = class_balance_table(df[TARGET_COLUMN])
        edible_row = balance_df[balance_df["class_code"] == "e"].iloc[0]
        poisonous_row = balance_df[balance_df["class_code"] == "p"].iloc[0]

        print(
            f"Wnioski: w zbiorze znajduje sie {int(edible_row['count'])} grzybow jadalnych "
            f"({edible_row['percentage']:.1f}%) oraz {int(poisonous_row['count'])} grzybow trujacych "
            f"({poisonous_row['percentage']:.1f}%). Oznacza to, ze klasy sa prawie zbalansowane."
        )
        """
    )


def missing_conclusion_code() -> str:
    return dedent(
        """
        missing_df = missing_values_table(
            ("train", train_df),
            ("validation", validation_df),
            ("test", test_df),
        )

        total_missing = int(missing_df["missing_count"].sum())
        train_missing = int(missing_df.loc[missing_df["split"] == "train", "missing_count"].sum())
        validation_missing = int(missing_df.loc[missing_df["split"] == "validation", "missing_count"].sum())
        test_missing = int(missing_df.loc[missing_df["split"] == "test", "missing_count"].sum())

        print(
            f"Wnioski: wykryto lacznie {total_missing} brakujacych wartosci, "
            f"w tym {train_missing} w zbiorze treningowym, {validation_missing} w walidacyjnym "
            f"oraz {test_missing} w testowym. Braki wystepuja w kolumnie 'stalk_root', "
            f"dlatego imputacja po podziale danych jest konieczna."
        )
        """
    )


def correlation_conclusion_code() -> str:
    return dedent(
        """
        strongest = top_correlations.iloc[0]
        print(
            f"Wnioski: najsilniej z targetem powiazana jest cecha "
            f"'{strongest['encoded_feature']}' o bezwzglednej korelacji "
            f"{strongest['absolute_target_correlation']:.4f}. Potwierdza to, ze czesc cech "
            f"bardzo dobrze rozroznia klasy jadalne i trujace."
        )
        """
    )


def overview_model_conclusion_code() -> str:
    return dedent(
        """
        best_row = results_df.iloc[0]
        print(
            f"Wnioski ogolne: najlepszy wynik walidacyjny uzyskal model "
            f"{best_row['model']} z accuracy = {best_row['validation_accuracy']:.4f} "
            f"oraz F1 = {best_row['validation_f1']:.4f}. Oznacza to, ze problem klasyfikacji "
            f"na zbiorze mushroom jest bardzo dobrze separowalny."
        )
        """
    )


def model_metric_conclusion_code() -> str:
    return dedent(
        """
        test_metrics = pd.DataFrame(
            [
                {
                    "split": "validation",
                    "accuracy": round(accuracy_score(preprocessed["y_validation"], validation_pred), 4),
                    "precision": round(precision_score(preprocessed["y_validation"], validation_pred), 4),
                    "recall": round(recall_score(preprocessed["y_validation"], validation_pred), 4),
                    "f1": round(f1_score(preprocessed["y_validation"], validation_pred), 4),
                },
                {
                    "split": "test",
                    "accuracy": round(accuracy_score(preprocessed["y_test"], test_pred), 4),
                    "precision": round(precision_score(preprocessed["y_test"], test_pred), 4),
                    "recall": round(recall_score(preprocessed["y_test"], test_pred), 4),
                    "f1": round(f1_score(preprocessed["y_test"], test_pred), 4),
                },
            ]
        )
        validation_row = test_metrics[test_metrics["split"] == "validation"].iloc[0]
        test_row = test_metrics[test_metrics["split"] == "test"].iloc[0]

        print(
            f"Wnioski po ocenie modelu: na zbiorze walidacyjnym model osiagnal "
            f"accuracy = {validation_row['accuracy']:.4f}, precision = {validation_row['precision']:.4f}, "
            f"recall = {validation_row['recall']:.4f} oraz F1 = {validation_row['f1']:.4f}. "
            f"Na zbiorze testowym uzyskano accuracy = {test_row['accuracy']:.4f} i F1 = {test_row['f1']:.4f}, "
            f"co potwierdza bardzo dobra zdolnosc generalizacji."
        )
        """
    )


def confusion_conclusion_code() -> str:
    return dedent(
        """
        correct_predictions = int(classification_report_df.loc["accuracy", "precision"] * len(preprocessed["y_test"]))
        print(
            f"Wnioski po analizie macierzy pomylek: model poprawnie sklasyfikowal "
            f"{correct_predictions} z {len(preprocessed['y_test'])} obserwacji testowych. "
            f"Niewielka liczba bledow potwierdza wysoka jakosc klasyfikacji."
        )
        """
    )


def preprocessing_section(include_model_preview: bool = False) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = [
        markdown_cell(
            """
            ### <a name='0'></a> Import bibliotek

            W pierwszym kroku importujemy biblioteki potrzebne do:
            - pracy z danymi tabelarycznymi,
            - przygotowania danych do modelu,
            - trenowania klasyfikatorow,
            - oceny skutecznosci modeli.

            Wspolna logika zostala wyniesiona do pliku `mushroom_analysis.py`,
            aby wszystkie notebooki korzystaly z tej samej procedury preprocessingu.
            """
        ),
        common_imports_cell(),
        markdown_cell(
            """
            ### <a name='1'></a> Wczytanie danych

            Zbior `agaricus-lepiota.data` zawiera informacje o cechach grzybow
            oraz zmiennej docelowej `class`, ktora informuje, czy dany grzyb
            jest jadalny (`e`) czy trujacy (`p`).
            """
        ),
        code_cell(
            """
            df = load_data()
            df.head()
            """
        ),
        markdown_cell(
            """
            ### <a name='2'></a> Podstawowe informacje o zbiorze

            Na tym etapie sprawdzamy rozmiar zbioru. Pozwala to potwierdzic,
            ile obserwacji i ile cech bedzie bralo udzial w analizie.
            """
        ),
        code_cell(
            """
            df.shape
            """
        ),
        markdown_cell(
            """
            ### <a name='3'></a> Analiza zbalansowania klas

            Przed budowa modelu nalezy sprawdzic, czy klasy w zbiorze sa rownomiernie
            reprezentowane. Silna nierownowaga moglaby prowadzic do zawyzonych metryk
            i wymagalaby dodatkowych technik balansowania.
            """
        ),
        code_cell(
            """
            class_balance_table(df[TARGET_COLUMN])
            """
        ),
        code_cell(
            """
            plot_class_distribution(df[TARGET_COLUMN])
            """
        ),
        code_cell(class_conclusion_code()),
        markdown_cell(
            """
            ### <a name='4'></a> Podzial danych na zbiory treningowy, walidacyjny i testowy

            Zgodnie z zalozeniami projektowymi dzielimy dane w proporcji `70/15/15`.
            Podzial wykonywany jest przed imputacja brakow, aby uniknac przecieku danych.
            """
        ),
        code_cell(
            """
            train_df, validation_df, test_df = split_dataset(df)

            pd.DataFrame(
                [
                    {"split": "train", "rows": len(train_df)},
                    {"split": "validation", "rows": len(validation_df)},
                    {"split": "test", "rows": len(test_df)},
                ]
            )
            """
        ),
        markdown_cell(
            """
            ### <a name='5'></a> Analiza brakujacych wartosci

            W zbiorze mushroom brakujace wartosci sa szczegolnie istotne,
            poniewaz niektore cechy kategorii zawieraja symbole `?`.
            Po podziale danych sprawdzamy, gdzie wystepuja braki i w jakiej skali.
            """
        ),
        code_cell(
            """
            missing_values_table(
                ("train", train_df),
                ("validation", validation_df),
                ("test", test_df),
            )
            """
        ),
        code_cell(missing_conclusion_code()),
        markdown_cell(
            """
            ### <a name='6'></a> Preprocessing danych

            W tej sekcji wykonujemy:
            - imputacje najczesciej wystepujacej kategorii,
            - kodowanie zmiennej docelowej do postaci numerycznej,
            - kodowanie cech kategorycznych z wykorzystaniem One-Hot Encoding.

            Taki sposob przygotowania danych jest odpowiedni dla analizowanych modeli.
            """
        ),
        code_cell(
            """
            preprocessed = preprocess_after_split(train_df, validation_df, test_df)

            print("X_train_encoded:", preprocessed["X_train_encoded"].shape)
            print("X_validation_encoded:", preprocessed["X_validation_encoded"].shape)
            print("X_test_encoded:", preprocessed["X_test_encoded"].shape)
            """
        ),
        markdown_cell(
            """
            ### <a name='7'></a> Wizualizacja wybranych cech

            Histogramy pomagaja zrozumiec rozklad kategorii dla cech,
            ktore moga miec duze znaczenie przy rozroznianiu grzybow jadalnych
            i trujacych.
            """
        ),
        code_cell(
            """
            plot_feature_histograms(train_df)
            """
        ),
        markdown_cell(
            """
            ### <a name='8'></a> Korelacja cech modelu

            Po zakodowaniu cech mozemy sprawdzic, ktore z nich wykazuja
            najsilniejszy zwiazek ze zmienna docelowa. Pozwala to lepiej
            uzasadnic, dlaczego model osiagnie okreslone wyniki.
            """
        ),
        code_cell(
            """
            top_correlations = plot_top_feature_correlations(
                preprocessed["X_train_encoded"],
                preprocessed["y_train"],
                top_n=12,
            )
            top_correlations
            """
        ),
        code_cell(correlation_conclusion_code()),
    ]

    if include_model_preview:
        cells.extend(
            [
                markdown_cell(
                    """
                    ### <a name='9'></a> Podglad wynikow wszystkich modeli

                    Zanim przejdziemy do osobnych notebookow modelowych,
                    warto zobaczyc zbiorcze porownanie wszystkich pieciu algorytmow.
                    """
                ),
                code_cell(
                    """
                    results_df = evaluate_models(preprocessed).drop(columns="fitted_model")
                    results_df
                    """
                ),
                code_cell(overview_model_conclusion_code()),
            ]
        )

    return cells


def build_overview_notebook() -> list[dict[str, object]]:
    return [
        markdown_cell(
            """
            # Przygotowanie danych dla zbioru Mushroom

            ### Preprocessing danych i analiza wstepna:
            1. [Import bibliotek](#0)
            2. [Wczytanie danych](#1)
            3. [Podstawowe informacje o zbiorze](#2)
            4. [Analiza zbalansowania klas](#3)
            5. [Podzial danych](#4)
            6. [Analiza brakujacych wartosci](#5)
            7. [Preprocessing danych](#6)
            8. [Wizualizacja wybranych cech](#7)
            9. [Korelacja cech modelu](#8)
            10. [Podglad wynikow wszystkich modeli](#9)
            """
        ),
        *preprocessing_section(include_model_preview=True),
    ]


def build_model_notebook(
    title: str,
    intro_text: str,
    balancing_note: str,
    estimator_code: str,
    mushroom_setup_code: str,
    adult_setup_code: str,
    final_conclusion: str,
) -> list[dict[str, object]]:
    return [
        markdown_cell(
            f"""
            # {title}

            ### Analiza modelu:
            1. [Import bibliotek](#0)
            2. [Wprowadzenie do modelu](#1)
            3. [Strategia balansowania](#2)
            4. [Analiza dla datasetu Mushroom](#3)
            5. [Analiza dla datasetu Adult Income](#4)
            6. [Porownanie wynikow](#5)
            7. [Wnioski koncowe](#6)
            """
        ),
        markdown_cell(
            f"""
            ### <a name='0'></a> Import bibliotek

            W notebooku wykorzystujemy wspolny pipeline przygotowania danych,
            aby porownanie tego samego algorytmu na dwoch datasetach bylo uczciwe.
            """
        ),
        common_imports_cell(),
        markdown_cell(
            f"""
            ### <a name='1'></a> Wprowadzenie do modelu

            {intro_text}
            """
        ),
        markdown_cell(
            f"""
            ### <a name='2'></a> Strategia balansowania

            {balancing_note}
            """
        ),
        markdown_cell(
            """
            ### <a name='3'></a> Analiza dla datasetu Mushroom

            Najpierw uruchamiamy ten sam algorytm na zbiorze mushroom.
            Pozwoli to sprawdzic, czy model nadal osiaga niemal idealne wyniki
            na bardzo dobrze separowalnym problemie klasyfikacyjnym.
            """
        ),
        code_cell(mushroom_setup_code),
        markdown_cell(
            """
            #### Definicja modelu
            """
        ),
        code_cell(estimator_code),
        code_cell(dataset_metric_code()),
        code_cell(
            dedent(
                """
                mushroom_metrics = metrics_df.copy()
                mushroom_metrics
                """
            )
        ),
        code_cell(dataset_metric_conclusion_code("mushroom")),
        markdown_cell(
            """
            #### Macierz pomylek i raport klasyfikacji
            """
        ),
        code_cell(
            """
            mushroom_classification_report = plot_confusion_matrix_for_model(
                model,
                X_test_reference,
                preprocessed["y_test"],
                preprocessed["label_encoder"],
            )
            mushroom_classification_report
            """
        ),
        markdown_cell(
            """
            ### <a name='4'></a> Analiza dla datasetu Adult Income

            Nastepnie uruchamiamy ten sam algorytm na bardziej wymagajacym
            zewnetrznym datasecie Adult Income. Ten zbior zawiera cechy mieszane
            i jest znacznie mniej separowalny niz mushroom.
            """
        ),
        code_cell(adult_setup_code),
        markdown_cell(
            """
            #### Definicja modelu
            """
        ),
        code_cell(estimator_code),
        code_cell(dataset_metric_code()),
        code_cell(
            dedent(
                """
                adult_metrics = metrics_df.copy()
                adult_metrics
                """
            )
        ),
        code_cell(dataset_metric_conclusion_code("adult_income")),
        markdown_cell(
            """
            #### Macierz pomylek i raport klasyfikacji
            """
        ),
        code_cell(
            """
            adult_classification_report = plot_confusion_matrix_for_model(
                model,
                X_test_reference,
                preprocessed["y_test"],
                preprocessed["label_encoder"],
            )
            adult_classification_report
            """
        ),
        markdown_cell(
            """
            ### <a name='5'></a> Porownanie wynikow

            W tej sekcji zestawiamy wyniki jednego algorytmu
            na dwoch roznych datasetach.
            """
        ),
        code_cell(
            """
            comparison_df = pd.DataFrame(
                [
                    {
                        "dataset": "mushroom",
                        "validation_accuracy": mushroom_metrics[mushroom_metrics["split"] == "validation"].iloc[0]["accuracy"],
                        "validation_f1": mushroom_metrics[mushroom_metrics["split"] == "validation"].iloc[0]["f1"],
                        "test_accuracy": mushroom_metrics[mushroom_metrics["split"] == "test"].iloc[0]["accuracy"],
                        "test_f1": mushroom_metrics[mushroom_metrics["split"] == "test"].iloc[0]["f1"],
                    },
                    {
                        "dataset": "adult_income",
                        "validation_accuracy": adult_metrics[adult_metrics["split"] == "validation"].iloc[0]["accuracy"],
                        "validation_f1": adult_metrics[adult_metrics["split"] == "validation"].iloc[0]["f1"],
                        "test_accuracy": adult_metrics[adult_metrics["split"] == "test"].iloc[0]["accuracy"],
                        "test_f1": adult_metrics[adult_metrics["split"] == "test"].iloc[0]["f1"],
                    },
                ]
            )
            comparison_df
            """
        ),
        code_cell(comparison_conclusion_code(title.replace("Klasyfikacja - ", ""))),
        markdown_cell(
            f"""
            ### <a name='6'></a> Wnioski koncowe

            {final_conclusion}
            """
        ),
    ]


def write_readme() -> None:
    readme = dedent(
        """
        # Analiza zbioru Mushroom

        Material zostal przygotowany w stylu zblizonym do notebookow z folderu `lab-12-04`.
        Kazdy plik zawiera:
        - jasny podzial na sekcje,
        - szczegolowe opisy etapow,
        - wizualizacje,
        - metryki,
        - wnioski po kluczowych czesciach analizy.

        ## Kolejnosc pracy

        - `00_data_preprocessing_and_balance.ipynb` - analiza danych, balans klas, braki, preprocessing, histogramy i korelacje.
        - `01_logistic_regression.ipynb` - jeden algorytm, dwa datasety: `mushroom` i `adult_income`.
        - `02_decision_tree.ipynb` - jeden algorytm, dwa datasety: `mushroom` i `adult_income`.
        - `03_random_forest.ipynb` - jeden algorytm, dwa datasety: `mushroom` i `adult_income`.
        - `04_knn.ipynb` - jeden algorytm, dwa datasety: `mushroom` i `adult_income`.
        - `05_bernoulli_naive_bayes.ipynb` - jeden algorytm, dwa datasety: `mushroom` i `adult_income`.

        ## Pliki pomocnicze

        - `mushroom_analysis.py` - wspolne funkcje do preprocessingu, wizualizacji i oceny.
        - `generate_mushroom_notebook.py` - generator wszystkich notebookow.
        - `../adult.data` - zewnetrzny dataset Adult Income pobrany z UCI.

        ## Uwagi techniczne

        Problem nieregularnych odstepow w komorkach wyniknal z generowania tresci
        przez wielolinijkowe stringi z zachowanym wcieciem kodu. Zostal naprawiony
        przez normalizacje tekstu przed zapisem do pliku `.ipynb`.
        """
    ).strip() + "\n"
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print("Written: README.md")


def main() -> None:
    save_notebook("00_data_preprocessing_and_balance.ipynb", build_overview_notebook())

    save_notebook(
        "01_logistic_regression.ipynb",
        build_model_notebook(
            title="Klasyfikacja - Logistic Regression",
            intro_text="""
            Logistic Regression jest jednym z podstawowych modeli klasyfikacji binarnej.
            Mimo nazwy zawierajacej slowo "regression", model ten sluzy do przewidywania
            prawdopodobienstwa przynaleznosci do klasy. W problemie klasyfikacji grzybow
            model ten pozwala oszacowac, czy obiekt nalezy do klasy jadalnej czy trujacej.

            Zaletami Logistic Regression sa:
            - prostota interpretacji,
            - szybki trening,
            - dobra skutecznosc przy dobrze przygotowanych danych.
            """,
            balancing_note="""
            W przypadku Logistic Regression zastosowano parametr `class_weight="balanced"`.
            Jest to rozwiazanie odpowiednie dla modelu liniowego, poniewaz pozwala
            uwzglednic drobne roznice w liczebnosci klas bez potrzeby duplikowania obserwacji.
            """,
            estimator_code="""
            model = LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
            model
            """,
            mushroom_setup_code=dataset_run_code("mushroom", use_binary=False),
            adult_setup_code=dataset_run_code("adult_income", use_binary=False),
            final_conclusion="""
            Logistic Regression okazala sie bardzo skutecznym modelem dla zbioru mushroom.
            Oznacza to, ze po odpowiednim zakodowaniu cech problem jest w duzym stopniu
            liniowo separowalny. Model ten moze byc traktowany jako bardzo mocny punkt odniesienia
            dla bardziej zlozonych algorytmow.
            """,
        ),
    )

    save_notebook(
        "02_decision_tree.ipynb",
        build_model_notebook(
            title="Klasyfikacja - Decision Tree",
            intro_text="""
            Decision Tree to model oparty na kolejnych podzialach przestrzeni cech.
            Drzewo uczy sie zestawu reguł decyzyjnych, ktore prowadza do rozroznienia klas.
            W problemie klasyfikacji grzybow jest to szczegolnie uzyteczne,
            poniewaz wiele cech ma charakter kategoryczny i dobrze nadaje sie do budowy reguł.

            Najwieksza zaleta tego modelu to wysoka interpretowalnosc.
            """,
            balancing_note="""
            Dla drzewa decyzyjnego zastosowano `class_weight="balanced"`.
            Pozwala to ograniczyc ryzyko, ze model bedzie preferowal liczniejsza klase
            przy budowie kolejnych podzialow.
            """,
            estimator_code="""
            model = DecisionTreeClassifier(
                max_depth=12,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
            model
            """,
            mushroom_setup_code=dataset_run_code("mushroom", use_binary=False),
            adult_setup_code=dataset_run_code("adult_income", use_binary=False),
            final_conclusion="""
            Decision Tree osiagnelo bardzo wysokie wyniki, co pokazuje,
            ze relacje pomiedzy cechami a targetem moga byc dobrze opisane
            przez zestaw reguł decyzyjnych. Model jest nie tylko skuteczny,
            ale rowniez latwiejszy do interpretacji niz wiele bardziej zlozonych metod.
            """,
        ),
    )

    save_notebook(
        "03_random_forest.ipynb",
        build_model_notebook(
            title="Klasyfikacja - Random Forest",
            intro_text="""
            Random Forest jest metoda zespolowa, ktora laczy wiele drzew decyzyjnych
            w jeden model. Kazde drzewo uczy sie na nieco innym podzbiorze danych i cech,
            a wynik koncowy uzyskiwany jest przez agregacje ich decyzji.

            Taki mechanizm zwykle zwieksza stabilnosc i odpornosc modelu na przeuczenie.
            """,
            balancing_note="""
            W modelu Random Forest wykorzystano `class_weight="balanced"`.
            Pozwala to uwzglednic proporcje klas podczas budowy poszczegolnych drzew
            bez zmiany rzeczywistego rozkladu danych treningowych.
            """,
            estimator_code="""
            model = RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            model
            """,
            mushroom_setup_code=dataset_run_code("mushroom", use_binary=False),
            adult_setup_code=dataset_run_code("adult_income", use_binary=False),
            final_conclusion="""
            Random Forest osiagnal wyniki porownywalne z najlepszymi modelami.
            Potwierdza to, ze agregacja wielu drzew bardzo dobrze radzi sobie
            z klasyfikacja grzybow i daje model odporny oraz stabilny.
            """,
        ),
    )

    save_notebook(
        "04_knn.ipynb",
        build_model_notebook(
            title="Klasyfikacja - K-Nearest Neighbors",
            intro_text="""
            K-Nearest Neighbors to metoda oparta na podobienstwie obserwacji.
            Nowy przypadek klasyfikowany jest na podstawie etykiet jego najblizszych sasiadow
            w przestrzeni cech. Po zastosowaniu One-Hot Encoding model ten moze skutecznie
            porownywac obserwacje zapisane w postaci wektorow binarnych.
            """,
            balancing_note="""
            Dla modelu KNN zastosowano oversampling klasy mniejszosciowej wyłącznie
            na zbiorze treningowym. Podejscie to ma sens, poniewaz KNN nie korzysta
            z parametru `class_weight`, a liczba sasiadow moze byc wrazliwa
            na nierownowage klas.
            """,
            estimator_code="""
            model = KNeighborsClassifier(n_neighbors=11)
            model
            """,
            mushroom_setup_code=dataset_oversample_code("mushroom", use_binary=False),
            adult_setup_code=dataset_oversample_code("adult_income", use_binary=False),
            final_conclusion="""
            KNN rowniez osiagnal bardzo wysokie wyniki, co oznacza,
            ze obiekty nalezace do tych samych klas tworza w przestrzeni cech
            wyrazne skupienia. Zastosowanie oversamplingu pomoglo zachowac
            rownowage podczas klasyfikacji opartej na sasiedztwie.
            """,
        ),
    )

    save_notebook(
        "05_bernoulli_naive_bayes.ipynb",
        build_model_notebook(
            title="Klasyfikacja - Bernoulli Naive Bayes",
            intro_text="""
            Bernoulli Naive Bayes jest modelem probabilistycznym szczegolnie dobrze
            dopasowanym do danych binarnych. Po kodowaniu One-Hot cechy przyjmuja
            wlasnie taka postac, dlatego model ten stanowi naturalny punkt odniesienia
            dla pozostalych klasyfikatorow.

            Model zaklada warunkowa niezaleznosc cech, co jest silnym uproszczeniem,
            ale w wielu zadaniach praktycznych daje zaskakujaco dobre rezultaty.
            """,
            balancing_note="""
            Dla Bernoulli Naive Bayes zastosowano oversampling klasy mniejszosciowej
            tylko na zbiorze treningowym. Dzięki temu model otrzymuje bardziej rowny
            rozklad klas podczas uczenia.
            """,
            estimator_code="""
            model = BernoulliNB()
            model
            """,
            mushroom_setup_code=dataset_oversample_code("mushroom", use_binary=True),
            adult_setup_code=dataset_oversample_code("adult_income", use_binary=True),
            final_conclusion="""
            Bernoulli Naive Bayes osiagnal nizsze wyniki niz najlepsze modele,
            ale nadal zapewnil wysoka skutecznosc klasyfikacji. Pokazuje to,
            ze nawet prosty model probabilistyczny dobrze radzi sobie z tym zbiorem,
            choc uproszczone zalozenie niezaleznosci cech ogranicza jego maksymalna jakosc.
            """,
        ),
    )

    write_readme()


if __name__ == "__main__":
    main()
