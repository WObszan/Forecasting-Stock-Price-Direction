# Forecasting Stock Price Direction

Powtarzalne eksperymenty z prognozowaniem kierunku cen AAPL, MSFT i GOOGL.
Projekt porównuje regresję logistyczną, Random Forest, SVM, XGBoost i soft voting
z prostymi baseline’ami, przy chronologicznym podziale danych i usuwaniu
nakładających się etykiet. Wyniki klasyfikacji nie są wynikiem strategii handlowej.

Repozytorium przeszło audyt metodologii i refaktoryzację. **Dawne wyniki nie
potwierdzają przewagi modelu**: wykryto m.in. wyciek danych w cechach i selekcji,
błędy AUC oraz nieprawidłowe przypisanie zwrotów do sygnałów.
Szczegóły, ograniczenia i plan dalszych badań: [docs/AUDIT.md](docs/AUDIT.md).
Przeprowadzone testy i nowe wyniki: [docs/VALIDATION.md](docs/VALIDATION.md).
Obsługa newsów, przegląd źródeł i zasady czasowe: [docs/NEWS.md](docs/NEWS.md).

## Uruchomienie

Python 3.11 lub nowszy. Polecenia wykonuj z katalogu głównego repozytorium.
Podstawowy eksperyment działa offline na zachowanych snapshotach cen.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m stock_direction --ticker AAPL
```

Domyślnie porównywane są LR, majority, AlwaysUp i losowy baseline, z treningiem
od 2010, pierwszym testem w 2016 i ostatnim w 2023. Wszystkie modele korzystają
z tych samych obserwacji. Każdy rok testowy ma osobny, rozszerzający się trening.
Nowe uruchomienie tworzy nowy katalog `reports/runs`.

Pełny zestaw i alternatywne etykiety:

```bash
python -m pip install -e '.[boosting,tuning]'
python -m stock_direction --ticker MSFT --models lr rf svm xgb ensemble majority always_up
python -m stock_direction --ticker GOOGL --target tbm --models lr rf majority always_up --horizon 5
python -m stock_direction --ticker AAPL --models lr --select-k 10 --tuning-trials 10
python -m stock_direction --help
```

Opcjonalnie można użyć `uv sync --locked --extra dev --extra boosting --extra tuning`
z dołączonym `uv.lock`, aby odtworzyć rozstrzygnięte wersje zależności.
Interaktywny przykład znajduje się w [notebooks/01_experiments.ipynb](notebooks/01_experiments.ipynb);
wymaga zainstalowanego pakietu i opcjonalnych zależności `notebooks`.

## Co jest oceniane

- **Binary:** wzrost ceny zamknięcia następnej sesji (`1`) albo spadek/brak zmiany
  (`0`). Ostatnia nieznana etykieta jest wykluczana. Domyślnie zachowujemy również
  małe ruchy. `--barrier-std 0.5` włącza ocenę warunkową po usunięciu małych
  *przyszłych* ruchów; nie jest to dostępna z góry reguła wyboru transakcji.
- **TBM:** pierwsze trafienie bariery na cenach zamknięcia w horyzoncie kilku sesji:
  strata `-1`, upływ czasu `0`, zysk `1`. Bariery bazują na historycznej zmienności
  z 20 sesji. To wariant close-based, bez oceny trafień intraday.
- **Cechy:** stopy zwrotu, historyczna zmienność, relacje OHLCV, RSI, MACD,
  położenie względem średnich i ekstremów oraz przyczynowy CUSUM/z-score.
  Domyślnie model korzysta z cen i wolumenu. Opcja `--news-csv` dodaje liczbę
  i sentyment historycznych nagłówków, dostępnych przed zamknięciem sesji.
- **Walidacja:** `Label_End` identyfikuje koniec zdarzenia; trening wyklucza
  etykiety sięgające początku roku testowego. Opcjonalny `--gap` usuwa dodatkowe
  wiersze treningu. Test nie jest skracany zależnie od długości przyszłych danych.
  Skalowanie, opcjonalna selekcja i strojenie są dopasowywane w treningu.
- **Metryki:** accuracy, balanced accuracy, precision/recall, macro F1 i ROC-AUC
  z prawdopodobieństw. Dla binary precision/recall dotyczą wzrostu; dla TBM są
  ważone licznością klas. AUC pozostaje puste, gdy w teście brakuje klasy.
  `comparison.csv` zawiera metryki pooled; `folds.csv` pozwala ocenić każdy rok.
- **Niepewność:** moving-block bootstrap accuracy (blok 20 obserwacji) i sparowany
  przedział różnicy względem majority, jeśli jest w zestawie modeli. Nie jest to
  korekta na wielokrotny wybór najlepszego eksperymentu.

Obecne CSV makro nie zawierają dat dostępności i historycznych rewizji, więc są
wyłączone z nowej oceny. Opcjonalny pipeline newsów używa VADER; FinBERT nie jest
obecnie wykorzystywany.
Historyczne ceny nie stanowią pełnej bazy point-in-time. Ocena ekonomiczna wymaga
osobnego backtestu z opóźnieniem wykonania, kosztami i regułami pozycji.

## Newsy i porównanie ich wpływu

Dodano opcjonalny pipeline newsów z VADER. Audyt publicznych zbiorów wykazał,
że nie zapewniają one zweryfikowanej historii AAPL/MSFT/GOOGL dla lat 2010–2023.
Działający pilot obejmuje **GOOGL i test 2020**, z datami wyodrębnionymi z URL
Nasdaq. To ograniczony eksperyment retrospektywny, z jawnymi założeniami dostępności.

```bash
python -m pip install -e '.[news,news-download]'
python -m stock_direction.news_cli fetch-fnspid
python -m stock_direction.news_cli prepare data/raw/news/fnspid_nasdaq100.csv --require-url-date
python -m stock_direction --ticker GOOGL --models lr rf majority always_up \
  --news-csv data/processed/news/vader_news.csv \
  --first-train-end-year 2019 --last-test-year 2020
```

Importer przypina rewizję podzbioru FNSPID Nasdaq-100 i wybiera nasze tickery.
Odrzucamy rekordy bez daty zakodowanej w URL Nasdaq; nie korzystamy z błędnych
czasów publikacji zapisanych w pliku. Pozostaje 4105 unikalnych nagłówków dla
Alphabet z lat 2019–2021. AAPL i MSFT nie mają w tej kopii wpisów spełniających
warunek daty URL. Nie zastępujemy tego braku fikcyjnym sentymentem.

`--news-csv` porównuje automatycznie **prices / prices_news** na tych samych
sesjach i etykietach. Raport zawiera sparowany przedział zmiany accuracy oraz
`news_coverage.csv`. Katalogi wyników mają nazwy np. `lr_prices` i `lr_prices_news`.

Reguła dostępności to koniec dnia z URL w Nowym Jorku + 24 h + 5 minut,
a następnie pierwsze późniejsze zamknięcie sesji. Kalendarz uwzględnia DST
oraz skrócone sesje. Daty URL są przybliżeniem dostępności, bez historii rewizji
artykułów. Do rozszerzenia badania na wszystkie spółki potrzebne są lepsze dane.
Szczegóły źródeł, odrzuceń i wyników: [docs/NEWS.md](docs/NEWS.md).
Notebook: [02_news_experiments.ipynb](notebooks/02_news_experiments.ipynb).

## Raporty i odtwarzalność

Każde uruchomienie zapisuje:

- `comparison.csv`: metryki i przedziały niepewności dla porównywanych modeli;
- `<model>/folds.csv`: daty graniczne, liczności i metryki każdego folda;
- `<model>/predictions.csv`: daty, etykiety, prognozy i prawdopodobieństwa nazwane
  rzeczywistymi klasami, np. `Proba_-1`, `Proba_0`, `Proba_1`;
- `<model>/selected_features.json`: cechy wybrane osobno dla każdego roku;
- `<model>/model_parameters.json`: parametry pipeline dla każdego roku, także po strojeniu;
- `metadata.json`: argumenty, seed, wersje pakietów, stan Git oraz SHA-256 wejścia i modułów kodu.

Katalog podany przez `--output-dir` musi być nowy, aby uniknąć nadpisania raportu.
Nowe raporty są ignorowane przez Git. Zachowane archiwum jest od nich oddzielone.

## Struktura

```text
src/stock_direction/       # dane, cechy, etykiety, modele, walidacja, strojenie, CLI
notebooks/
  01_experiments.ipynb     # przykład korzystający z aktualnego pakietu
  02_news_experiments.ipynb # pobieranie, audyt i porównanie cen z newsami
  legacy/                 # dawne notebooki i helpery; archiwum, nie aktualny pipeline
data/
  raw/prices/             # wejściowe snapshoty OHLCV
  raw/macro/              # archiwalne FRED, bez historii dostępności
  raw/sentiment/          # archiwalne VIX/EPU/headlines
  raw/news/               # nowe nagłówki z manifestem pochodzenia, ignorowane
  processed/news/         # oczyszczone nagłówki z sentymentem VADER, ignorowane
  interim/technical/      # dawne wskaźniki
  legacy/                 # dawne scalone tabele z cechami i etykietami
reports/
  legacy/                 # dawne wyniki, listy cech i wykresy
  runs/                   # nowe raporty, ignorowane przez Git
tests/                   # testy regresji i integracji
docs/                    # audyt i zachowana prezentacja
pyproject.toml            # instalacja i opcjonalne zależności
uv.lock                   # rozstrzygnięte wersje zależności
.github/workflows/        # CI
```

## Weryfikacja

```bash
pytest -q
ruff check .
```

Testy obejmują niezmienność cech po dopisaniu przyszłych danych, brak wpływu
etykiet testowych na selekcję, purging po datach zdarzeń, granice etykiet,
zgodność klas/prawdopodobieństw, AUC, bootstrap i pełne uruchomienie CLI.
Testy XGBoost/ensemble i Optuny są pomijane, jeśli nie zainstalowano dodatków.
CI instaluje te dodatki, aby również je sprawdzić.

Priorytety rozwoju: zamrożony protokół badawczy, porównanie z prostymi modelami,
realistyczny backtest, dane point-in-time i dopiero potem kolejne źródła cech.
Nie ustalono licencji repozytorium; wcześniejsza deklaracja MIT nie była poparta
plikiem LICENSE. Warunki użycia danych dostawców wymagają osobnego sprawdzenia.
