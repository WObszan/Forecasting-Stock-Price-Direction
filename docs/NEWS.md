# Newsy: implementacja i audyt danych

Dodano opcjonalny pipeline nagłówków z sentymentem VADER oraz automatyczne
porównanie `prices` / `prices_news` na identycznych datach i etykietach.
**Działający pilot obejmuje GOOGL i test 2020. Nie mamy zatwierdzonego zbioru
newsów dla całej historii AAPL/MSFT/GOOGL z lat 2010–2023.**

## Co faktycznie sprawdzono — 2026-11-23

| Źródło | Wynik sprawdzenia | Zastosowanie |
| --- | --- | --- |
| [FNSPID autorów](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) i [pliki źródłowe](https://huggingface.co/datasets/Zihan1004/FNSPID/tree/main/Stock_news) | Tematycznie pasuje do badania cen z tekstem. CSV mają ok. 5,73 GB i 23,2 GB; pełny viewer nie działa z powodu błędu schematu. Deklarowany okres zbioru nie dowodzi kompletnej historii newsów każdej spółki. | Nie użyto niezweryfikowanych czasów publikacji. |
| [Lustro FNSPID_external](https://huggingface.co/datasets/sabareesh88/FNSPID_external) | Skan wszystkich 13 shardów zwrócił 3431 pasujących rekordów, bez MSFT. Sprawdzone późniejsze grupy mają puste Stock_symbol. Normalizacja wielkości liter nie odzyskała historii. | Odrzucone jako wejście głównego badania. |
| [FNSPID Nasdaq-100](https://huggingface.co/datasets/benstaf/FNSPID-filtered-nasdaq-100) | Wybrano 29759 rekordów naszych tickerów z dwóch shardów. Preview ma przykłady daty 2023-12-16 przy opisie premarketu 2023-12-19. Zapisany Date nie jest wiarygodnym zegarem publikacji. | Zachowano wyłącznie URL Nasdaq zakończone datą; zignorowano Date pliku. |
| [Kaggle: Daily Financial News for 6000+ Stocks](https://www.kaggle.com/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests/metadata) | Pobrano oryginalny plik processed i raw. Dla naszych tickerów dały odpowiednio 3263 i 3219 rekordów, bez MSFT; historia AAPL zaczyna się dopiero w marcu 2020. Autor rozróżnia precyzję dat poszczególnych plików. | Nie użyto do wspólnej walidacji trzech spółek. Pomocniczy importer `fetch-benzinga` zachowano. |
| [GDELT](https://www.gdeltproject.org/data.html) | Publiczne archiwum metadanych o dużym wolumenie. Wymaga mapowania organizacji na spółki oraz audytu znaczenia timestampów. | Kandydat na osobny projekt pozyskania danych; nie został podłączony do modeli. |
| [Alpha Vantage NEWS_SENTIMENT](https://www.alphavantage.co/documentation/#news-sentiment) | API ma ticker, zakres czasu i sentyment; dokumentacja oznacza endpoint jako premium i podaje limit 1000 wyników odpowiedzi. | Wymaga dostępu użytkownika i sprawdzenia pokrycia. Nie potwierdzono historii 2010–2023. |
| [Archiwum Bloomberg/Reuters](https://github.com/philipperemy/financial-news-dataset) | Autor informuje o wycofaniu publicznego udostępniania ze względu na prawa autorskie. | Nie wykorzystano. |

To wyniki inspekcji konkretnych kopii, a nie twierdzenie, że dla tych spółek
nie istnieją żadne lepsze archiwa. Audyt i hashe odrzuconych wejść są w
[news_source_audit.json](../reports/news_source_audit.json).

Warunki użycia FNSPID są opisane niespójnie: [karta HF](https://huggingface.co/datasets/Zihan1004/FNSPID/blob/main/README.md)
wskazuje CC BY-NC 4.0, a README autorów na GitHub zawiera późniejszą informację
o uwolnieniu użycia badawczego i komercyjnego. Nie nadajemy tekstom licencji
kodu projektu. Pobrane nagłówki pozostają lokalnie, poza Git.

## Przyjęty podzbiór i znaczenie dat

W podzbiorze Nasdaq-100 znaleziono 4106 rekordów z URL postaci
`https://www.nasdaq.com/articles/...-YYYY-MM-DD`. Po usunięciu jednego duplikatu
pozostało **4105 nagłówków Alphabet**, z datami dostępności w latach 2019–2021:
1330 / 1555 / 1220. GOOG i GOOGL połączono na poziomie emitenta.
Pozostałe 25653 rekordy odrzucono z powodu braku obsługiwanego URL z datą.
W 92 zachowanych rekordach dzień z URL różnił się od dnia w polu Date źródła.

AAPL i MSFT nie mają w tej konkretnej kopii wpisów spełniających warunek daty
URL. CLI zgłasza brak pokrycia zamiast tworzyć dla nich pozorny wynik newsów.
Także dla GOOGL jest to podzbiór obserwowanych artykułów, nie kompletny strumień
wszystkich wiadomości o spółce.

Dzień zakodowany w adresie traktujemy jako **przybliżenie dnia publikacji**.
Nie zweryfikowano historycznego pierwszego udostępnienia każdego tekstu ani
jego późniejszych rewizji. Reguła dostępności to koniec tego dnia w
`America/New_York`, potem 24 h opóźnienia i 5 min na przetworzenie. Artykuł
trafia do pierwszej sesji zamykanej ściśle później. Opóźnienie nie dowodzi,
że wszystkie ewentualne błędy dat i rewizje tekstu zostały wyeliminowane.

Kalendarz uwzględnia DST, święta i skrócone sesje. W bibliotece
`exchange_calendars` używamy XNYS dla regularnych sesji amerykańskich,
stosowanego również dla NASDAQ. Zobacz [harmonogram Nasdaq](https://www.nasdaq.com/market-activity/stock-market-holiday-schedule).
Daty cen spoza kalendarza są odrzucane. Nowe artykuły nie uzupełniają danych
wcześniejszych sesji.

## Implementacja i uruchomienie

```bash
python -m pip install -e '.[news,news-download]'
python -m stock_direction.news_cli fetch-fnspid
python -m stock_direction.news_cli prepare data/raw/news/fnspid_nasdaq100.csv --require-url-date
python -m stock_direction --ticker GOOGL --models lr rf svm xgb ensemble majority always_up random \
  --news-csv data/processed/news/vader_news.csv \
  --first-train-end-year 2019 --last-test-year 2020
```

Do pełnego zestawu modeli potrzebny jest także dodatek `boosting`.
Pobieranie przypina rewizję `benstaf/FNSPID-filtered-nasdaq-100` i odczytuje
kolumny nagłówków przez HTTP ranges. Nie zapisuje treści artykułów ani podglądu
pierwszych wierszy jako rzekomo pełnego zbioru. Importer normalizuje case symboli,
odrzuca puste tytuły i niepoprawne daty, usuwa powtarzające się nagłówki w dniu
oraz zachowuje oryginalną datę w `Source_Published_At`.

Znany input FNSPID z manifestem nie może być przygotowany bez
`--require-url-date`. Ogólny importer dla innych CSV wspiera `--source-timezone`,
`--trust-intraday-times` i `--delay-hours`; weryfikacja ich zegara pozostaje
warunkiem poprawnej interpretacji. Nie użyto opcji zaufania godzinom w pilocie.

Sentyment liczy [VADER](https://github.com/cjhutto/vaderSentiment), zamrożony
słownik i reguły bez dopasowania do etykiet cenowych. To ogólny baseline,
nie model wyspecjalizowany w finansach. Zachowujemy wersję i hash słownika.
Osiem cech obejmuje liczbę i obecność artykułów, średni score, jego zmienność,
udziały pozytywne/negatywne oraz średnie sentymentu i liczby z siedmiu sesji.
`news_present` odróżnia brak obserwowanych artykułów od średniej neutralnej.
Brak artykułów w archiwum nie dowodzi braku wydarzeń.

Dla każdego modelu powstają warianty `prices` i `prices_news`, z tymi samymi
datami, etykietami i seedem. Raport sprawdza zgodność kalendarzy i liczy
sparowany bootstrap różnicy accuracy. Nie przeprowadza nowego doboru modeli
na podstawie wyników pilota.

Notebook [02_news_experiments.ipynb](../notebooks/02_news_experiments.ipynb)
korzysta z tych samych funkcji; archiwalne notebooki nie są potrzebne.
Katalogi danych i raportów nie są nadpisywane.

## Wyniki i ograniczenia pilota

Sprawdzono GOOGL, osiem modeli, dwa zestawy cech i cele binary/TBM: 32 konfiguracje,
po 253 prognozy z 2020 r. każda. Trening zaczyna się w 2010 i kończy w 2019,
a historyczne newsy dostępne są w treningu dopiero od 2019.

Dla ensemble binary accuracy zmieniło się z 51,78% do 52,57%, przy baseline
AlwaysUp 56,92%. Dla ensemble TBM zmieniło się z 49,01% do 47,43%.
Sparowane 95% przedziały zmiany accuracy zawierają zero dla wszystkich
badanych modeli ML. Pilot nie potwierdza poprawy dzięki newsom. Jeden rok,
obejmujący pandemię, nie wystarcza do uogólnienia tego wyniku.

Pełne metryki: [news_validation_summary.csv](../reports/news_validation_summary.csv).
Manifesty: [news_validation_metadata.json](../reports/news_validation_metadata.json).
Pokrycie zaakceptowanych newsów: [news_coverage_by_year.csv](../reports/news_coverage_by_year.csv).
Pełne prognozy, foldy i pokrycie sesji są lokalnie w
`reports/runs/validated_news_googl_<binary|tbm>_2020/`.

Przeszły 62 testy, również dla przyczynowego przypisania wiadomości do sesji,
DST, skróconych sesji, deduplikacji, opóźnień, wymaganego źródła daty i
pełnego wariantu porównawczego CLI. Biblioteka kalendarza emituje ostrzeżenia
o przyszłej zmianie API pandas; testy przechodzą bez ich globalnego ukrywania.

Dalszy etap to pozyskanie niezależnie sprawdzonego archiwum dla wszystkich
trzech emitentów, weryfikacja próbek artykułów i historii aktualizacji,
a następnie wieloletnia ocena oraz osobne porównanie count-only i sentymentu.
Przed wyborem płatnego dostawcy trzeba potwierdzić pokrycie konkretnych tickerów
w wymaganych latach i możliwość odtworzenia informacji dostępnej w danym dniu.

FinBERT nie został uruchomiony w tej zmianie. [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
jest finansowym modelem sentymentu, ale użycie późniejszego enkodera na dawnych
danych byłoby badaniem retrospektywnym. Także obecny słownik VADER traktujemy
jako zamrożone współczesne narzędzie, bez deklaracji, że identyczna wersja była
wdrożona w badanym roku. Wynik klasyfikacji nadal nie jest wynikiem strategii
z kosztami i wykonalnym momentem wykonania transakcji.
