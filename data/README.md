# Dane

- `raw/prices`: zachowane snapshoty OHLCV AAPL/MSFT/GOOGL; wejście do CLI.
- `raw/macro`: archiwalne serie FRED; brak historii publikacji i rewizji,
  dlatego nie są wejściem domyślnego modelu.
- `raw/sentiment`: archiwalne VIX/EPU i headlines; nie stanowią zweryfikowanego
  zbioru historycznego sentymentu NLP.
- `raw/news`: pobrane historyczne nagłówki z manifestem źródła i hashem; ignorowany przez Git.
- `processed/news`: oczyszczone nagłówki, konserwatywny czas dostępności i sentyment
  VADER; ignorowane przez Git. Powstają przez `python -m stock_direction.news_cli prepare`; dla FNSPID wymagamy `--require-url-date`.
  Zatwierdzony podzbiór zawiera wyłącznie GOOGL; pokrycie opisano w `docs/NEWS.md`.
- `interim/technical`: zachowane techniczne wskaźniki z pierwotnego notebooka;
  nowy kod przelicza własne cechy od początku.
- `legacy`: stare scalone tabele z cechami i etykietami, zawierające problemy
  opisane w `docs/AUDIT.md`; wyłączone z aktualnego pipeline'u.

Wersjonowane ceny obejmują: AAPL/MSFT od 2000-01-03, GOOGL od 2004-08-19;
wszystkie trzy snapshoty kończą się 2026-01-16. CLI nie pobiera nowych danych.
Zakres walidacji jest oddzielnym parametrem; domyślny ostatni testowany rok to 2023.

Pliki zostały przeniesione bez zmiany zawartości; usunięto początkowe spacje
z nazw CSV. Dane cenowe są skorygowane przez dostawcę; pierwotny notebook
nie zachował pełnego manifestu pobrania. Nie należy przypisywać danym licencji
kodu; warunki redystrybucji dostawców wymagają osobnego ustalenia.
