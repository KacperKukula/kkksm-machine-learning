# Projekt analizy klasyfikacyjnej: Mushroom i Adult Income

**Autorzy opracowania:**  
- Kateryna Kolioglo 113577  
- Kacper Kukuła 113132  
- Kacper Zamiatała 113667  
- Szczepan Kurtek 113323  
- Mateusz Krówczyński 89531  
**Kontekst projektu:** porownawcza analiza wybranych algorytmow uczenia maszynowego
na dwoch datasetach o roznej charakterystyce i trudnosci klasyfikacji.


## Kolejnosc pracy

- `00_data_preprocessing_and_balance.ipynb` - analiza danych, balans klas, braki, preprocessing, histogramy i korelacje.
- `01_logistic_regression.ipynb` - jeden algorytm, dwa datasety: `mushroom` i `adult_income`.
- `02_decision_tree.ipynb` - jeden algorytm, dwa datasety: `mushroom` i `adult_income`.
- `03_random_forest.ipynb` - jeden algorytm, dwa datasety: `mushroom` i `adult_income`.
- `04_knn.ipynb` - jeden algorytm, dwa datasety: `mushroom` i `adult_income`.
- `05_bernoulli_naive_bayes.ipynb` - jeden algorytm, dwa datasety: `mushroom` i `adult_income`.

## Logika dokumentacji

Struktura projektu zostala podzielona na:
- notebook wprowadzajacy `00`, ktory opisuje dane, preprocessing i charakter problemu,
- notebooki `01-05`, z ktorych kazdy dotyczy jednego algorytmu
  analizowanego na dwoch datasetach,

Taki uklad odpowiada standardowi akademickiemu: oddziela przygotowanie danych,
uzasadnienie metodologiczne oraz czesc eksperymentalna.

## Pliki pomocnicze

- `mushroom_analysis.py` - wspolne funkcje do preprocessingu, wizualizacji i oceny.
- `generate_mushroom_notebook.py` - generator wszystkich notebookow.
- `../adult.data` - zewnetrzny dataset Adult Income pobrany z UCI.
- `EXPLANATIONS_PL.md` - pomocniczy opis notebookow do wykorzystania podczas prezentacji.

## Uwagi techniczne

Problem nieregularnych odstepow w komorkach wyniknal z generowania tresci
przez wielolinijkowe stringi z zachowanym wcieciem kodu. Zostal naprawiony
przez normalizacje tekstu przed zapisem do pliku `.ipynb`.
