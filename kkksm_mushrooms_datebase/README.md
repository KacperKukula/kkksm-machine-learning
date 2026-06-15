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
