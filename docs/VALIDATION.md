# Weryfikacja zmian — 2026-11-23

W lokalnym środowisku Python 3.13.18 przeszło **46 testów**, `ruff check .`,
kontrola formatowania i zgodności lockfile. Sprawdzono także składnię komórek
nowego notebooka i lokalne linki dokumentacji. Wszystkie **59 przeniesionych
plików** są zgodne bajt w bajt z oryginałami z `HEAD` sprzed refaktoryzacji.
Workflow CI dodano dla Python 3.11 i 3.13; nie był uruchamiany zdalnie w tym audycie.

Uruchomiono 48 konfiguracji: trzy spółki × dwa rodzaje etykiet × osiem modeli,
po osiem rocznych foldów każda (384 treningi zewnętrzne, nie licząc wewnętrznych
składowych ensemble/kalibracji). Każda konfiguracja ocenia 2012 obserwacji z lat
2016–2023. Trening rozszerza się od 2010, bez dodatkowego gap; purging korzysta
z dat zdarzeń. Cechy mają historię potrzebną do rozgrzewki sprzed 2010.
Brak strojenia i selekcji K w tych 48 konfiguracjach; seed 42, TBM horizon 5,
binary bez filtrowania przyszłych ruchów. Wejściem są wyłącznie snapshoty OHLCV.

## Wyniki klasyfikacji binarnej

| Spółka | LR accuracy | Ensemble accuracy | AlwaysUp / majority accuracy | Ensemble ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| AAPL | 50,70% | 51,69% | 53,43% | 0,507 |
| GOOGL | 52,04% | 52,14% | 53,13% | 0,510 |
| MSFT | 53,83% | 53,08% | 53,93% | 0,511 |

W tym protokole żaden z badanych modeli ML nie przekroczył accuracy AlwaysUp
na żadnej z trzech spółek. Dla ensemble wszystkie sparowane 95% przedziały
różnicy accuracy względem majority zawierają zero. Te wyniki nie uzasadniają
deklaracji o stabilnej przewadze ensemble.

## Wyniki TBM

| Spółka | LR accuracy | Ensemble accuracy | Majority accuracy | AlwaysUp accuracy | Ensemble ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| AAPL | 43,64% | 40,71% | 40,76% | 42,99% | 0,549 |
| GOOGL | 41,40% | 41,60% | 36,68% | 39,81% | 0,573 |
| MSFT | 42,25% | 40,36% | 41,65% | 41,65% | 0,555 |

GOOGL/TBM daje obiecujący wynik względem majority; dla ensemble sparowany
przedział różnicy wynosi około [+1,04; +9,10] punktu procentowego. Majority jest
jednak słabsze od AlwaysUp w tym eksperymencie. Nie wykonano korekty na wiele
spółek, modeli i zadań ani odrębnego testu ekonomicznego. To obserwacja do dalszej
weryfikacji, bez podstaw do deklarowania rentowności. TBM ma inne klasy i bazową
trudność zadania niż binary, więc nie należy porównywać ich accuracy wprost.

Przy porównywaniu ROC-AUC pooled należy pamiętać, że majority może zmieniać
przewidywaną klasę między latami. Dlatego jego AUC dla połączonych foldów może
różnić się od 0,5 mimo stałej prognozy wewnątrz każdego roku.

## Artefakty i odtworzenie

Pełne metryki, również balanced accuracy, macro F1 i przedziały bootstrap:
[reports/validation_summary.csv](../reports/validation_summary.csv).
Argumenty, wersje bibliotek, hashe wejściowych danych i manifest kodu każdego
uruchomienia: [reports/validation_metadata.json](../reports/validation_metadata.json).
Stan Git w manifestach jest oznaczony jako dirty, ponieważ zmiany nie zostały
zacommitowane. Historyczny commit w metadanych nie obejmuje nowego kodu;
osobne hashe modułów identyfikują użyte źródła.

Pełne prognozy, parametry i foldy są lokalnie w
`reports/runs/final_<ticker>_<binary|tbm>/` (katalog ignorowany przez Git).
Poniższy przykład odtwarza jedną z sześciu konfiguracji wejścia; użyj nowego
katalogu wynikowego i powtórz dla pozostałych spółek/etykiet:

```bash
python -m stock_direction --ticker AAPL --target binary \
  --models lr rf svm xgb ensemble majority always_up random \
  --output-dir reports/runs/reproduction_aapl_binary
```

Dodatkowo przeszło rzeczywiste uruchomienie MSFT/LR z wyborem 10 cech i jedną
próbą Optuny, z testem 2016. Jest to test działania ścieżki strojenia, a nie wynik
pełnej optymalizacji modelu. Testy automatyczne obejmują też wewnętrzną kalibrację
SVM, purged CV i zapis parametrów XGBoost zawierających wartości niefinitywne.

Najbliższy sensowny etap to zamrożenie protokołu, porównanie prostych modeli
z baseline’ami i sprawdzenie kosztów oraz wykonalności transakcji. Rozbudowę
ensemble/NLP/makro warto uzależnić od mierzalnej poprawy względem tych punktów
odniesienia. Dalsze propozycje i ograniczenia są w [audycie](AUDIT.md).
