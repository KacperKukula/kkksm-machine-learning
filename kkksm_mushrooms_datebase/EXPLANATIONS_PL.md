# Opis notebookow do prezentacji

Ten dokument zawiera praktyczne wyjasnienie, co dzieje sie w kazdym notebooku
oraz jak mozna o nim opowiedziec podczas prezentacji.

## 00_data_preprocessing_and_balance.ipynb

### Cel notebooka

Ten notebook pokazuje przygotowanie danych dla zbioru `mushroom`.
Jest to etap wstepny przed budowa modeli klasyfikacyjnych.

### Co sie tu dzieje

1. Importowane sa biblioteki i funkcje pomocnicze.
2. Wczytywany jest zbior `agaricus-lepiota.data`.
3. Sprawdzany jest rozmiar zbioru danych.
4. Analizowany jest rozklad klas `edible` i `poisonous`.
5. Dane sa dzielone na:
   - zbior treningowy,
   - zbior walidacyjny,
   - zbior testowy.
6. Sprawdzane sa brakujace wartosci.
7. Po podziale wykonywany jest preprocessing:
   - imputacja brakow,
   - kodowanie cech kategorycznych,
   - przygotowanie danych do modeli.
8. Tworzone sa histogramy wybranych cech.
9. Analizowana jest korelacja zakodowanych cech z targetem.
10. Na koncu pokazywany jest zbiorczy podglad wynikow wszystkich modeli.

### Co powiedziec na prezentacji

"W tym notebooku przygotowujemy dane do analizy. Najpierw sprawdzamy,
czy klasy sa zbalansowane, potem dzielimy dane na train, validation i test.
Dopiero po podziale uzupelniamy braki, aby uniknac przecieku danych.
Nastepnie kodujemy cechy i sprawdzamy, ktore z nich sa najbardziej informacyjne."

### Najwazniejsze wnioski

- Zbior `mushroom` jest prawie zbalansowany.
- Braki danych wystepuja glownie w kolumnie `stalk_root`.
- Niektore cechy bardzo silnie rozrozniaja klasy, dlatego ten dataset jest stosunkowo latwy.

## 01_logistic_regression.ipynb

### Cel notebooka

Notebook pokazuje dzialanie algorytmu Logistic Regression na dwoch datasetach:
- `mushroom`,
- `adult_income`.

### Co sie tu dzieje

1. Przedstawiane jest krotkie wprowadzenie do Logistic Regression.
2. Opisana jest strategia balansowania klas: `class_weight="balanced"`.
3. Ten sam model jest uruchamiany najpierw na `mushroom`.
4. Liczone sa metryki:
   - accuracy,
   - precision,
   - recall,
   - F1-score.
5. Tworzona jest macierz pomylek oraz classification report dla `mushroom`.
6. Nastepnie identyczny algorytm uruchamiany jest na `adult_income`.
7. Znowu liczone sa metryki i tworzona jest macierz pomylek.
8. Na koncu wyniki z obu datasetow sa zestawiane w jednej tabeli.

### Co powiedziec na prezentacji

"Tutaj analizujemy jeden algorytm, czyli Logistic Regression, ale na dwoch roznych problemach.
Na zbiorze mushroom model dziala niemal idealnie, bo dane sa bardzo dobrze separowalne.
Na adult_income wyniki sa nizsze, co pokazuje, ze ten drugi dataset jest znacznie trudniejszy."

### Najwazniejsze wnioski

- Logistic Regression jest bardzo dobrym punktem odniesienia.
- Na `mushroom` model osiaga bardzo wysokie wyniki.
- Na `adult_income` wyniki sa bardziej realistyczne i pokazują, jak model zachowuje sie na trudniejszym zbiorze.

## 02_decision_tree.ipynb

### Cel notebooka

Notebook prezentuje dzialanie modelu Decision Tree na:
- `mushroom`,
- `adult_income`.

### Co sie tu dzieje

1. Wyjasniane jest, czym jest drzewo decyzyjne.
2. Opisane jest wykorzystanie `class_weight="balanced"`.
3. Model uruchamiany jest na `mushroom`.
4. Sprawdzane sa metryki i macierz pomylek.
5. Ten sam model uruchamiany jest na `adult_income`.
6. Ponownie analizowane sa metryki i raport klasyfikacji.
7. Wyniki sa porownywane miedzy datasetami.

### Co powiedziec na prezentacji

"Decision Tree buduje reguly decyzyjne na podstawie cech.
W przypadku danych o grzybach ten model dziala bardzo dobrze, bo wiele cech kategorycznych
mozna zamienic na czytelne reguly. Porownanie z adult_income pokazuje jednak,
ze nie kazdy problem jest tak prosty jak mushroom."

### Najwazniejsze wnioski

- Drzewo dobrze radzi sobie z cechami kategorycznymi.
- Na `mushroom` daje bardzo wysokie wyniki.
- Na `adult_income` pokazuje slabsza generalizacje niz na danych grzybowych.

## 03_random_forest.ipynb

### Cel notebooka

Notebook pokazuje dzialanie Random Forest na:
- `mushroom`,
- `adult_income`.

### Co sie tu dzieje

1. Opisany jest model Random Forest jako metoda zespolowa.
2. Pokazane jest wykorzystanie `class_weight="balanced"`.
3. Model trenowany jest na `mushroom`.
4. Liczone sa metryki i tworzona jest macierz pomylek.
5. Potem ten sam model trenowany jest na `adult_income`.
6. Wyniki obu uruchomien sa porownywane.

### Co powiedziec na prezentacji

"Random Forest laczy wiele drzew decyzyjnych i dzieki temu zwykle jest bardziej stabilny
od pojedynczego drzewa. W naszym projekcie bardzo dobrze radzi sobie na mushroom
i jednoczesnie daje sensowne wyniki na bardziej wymagajacym adult_income."

### Najwazniejsze wnioski

- Random Forest jest jednym z najmocniejszych modeli w tym projekcie.
- Na `mushroom` osiaga prawie idealne wyniki.
- Na `adult_income` utrzymuje dobre wyniki, co potwierdza jego stabilnosc.

## 04_knn.ipynb

### Cel notebooka

Notebook pokazuje dzialanie modelu K-Nearest Neighbors na:
- `mushroom`,
- `adult_income`.

### Co sie tu dzieje

1. Wyjasnione jest, ze KNN opiera sie na podobienstwie obserwacji.
2. Dla tego modelu stosowany jest oversampling klasy mniejszosciowej.
3. Model uruchamiany jest na `mushroom`.
4. Obliczane sa metryki i tworzona jest macierz pomylek.
5. To samo wykonywane jest dla `adult_income`.
6. Na koncu wyniki sa porownywane.

### Co powiedziec na prezentacji

"KNN klasyfikuje nowe obserwacje na podstawie najblizszych sasiadow.
Poniewaz ten model nie ma parametru `class_weight`, stosujemy oversampling
na zbiorze treningowym. Dzieki temu model ma bardziej wyrownane warunki uczenia."

### Najwazniejsze wnioski

- KNN dobrze dziala na danych, gdzie obiekty podobnych klas sa blisko siebie.
- Na `mushroom` model osiaga bardzo dobre wyniki.
- Na `adult_income` skutecznosc spada, bo problem jest bardziej zlozony.

## 05_bernoulli_naive_bayes.ipynb

### Cel notebooka

Notebook pokazuje dzialanie Bernoulli Naive Bayes na:
- `mushroom`,
- `adult_income`.

### Co sie tu dzieje

1. Wyjasniony jest probabilistyczny charakter modelu.
2. Opisane jest, dlaczego dla tego modelu przygotowywana jest binarna reprezentacja danych.
3. Stosowany jest oversampling klasy mniejszosciowej na zbiorze treningowym.
4. Model uruchamiany jest najpierw na `mushroom`.
5. Potem uruchamiany jest na `adult_income`.
6. Na koncu wyniki sa porownywane.

### Co powiedziec na prezentacji

"Bernoulli Naive Bayes najlepiej pasuje do danych binarnych.
Po kodowaniu One-Hot wiele cech ma wlasnie taka postac, dlatego ten model
jest ciekawym punktem odniesienia. Jest prostszy od innych modeli,
ale mimo to daje przyzwoite wyniki."

### Najwazniejsze wnioski

- Model probabilistyczny jest prostszy i ma mocne zalozenia.
- Na `mushroom` radzi sobie dobrze, ale gorzej niz najlepsze modele.
- Na `adult_income` pokazuje jeszcze wyrazniej ograniczenia uproszczonych zalozen.

## Jak opowiadac o calej strukturze projektu

### Najprostsze wyjasnienie

"Najpierw przygotowalismy dane i sprawdzilismy ich rozklad w notebooku 00.
Nastepnie dla kazdego algorytmu stworzylismy osobny notebook.
Kazdy z tych notebookow pokazuje, jak ten sam model zachowuje sie
na dwoch roznych datasetach: prostszym `mushroom` i trudniejszym `adult_income`.
Dzieki temu mozemy pokazac nie tylko jak dziala model,
ale tez jak zmienia sie jego skutecznosc w zaleznosci od charakteru danych."

### Najwazniejszy przekaz na prezentacji

- `mushroom` jest datasetem latwiejszym i lepiej separowalnym,
- `adult_income` jest datasetem bardziej realistycznym i trudniejszym,
- porownanie tych dwoch zbiorow pozwala lepiej ocenic prawdziwe mozliwosci algorytmow,
- bardzo wysokie wyniki na `mushroom` nie wynikaja z bledu, tylko ze specyfiki danych.
