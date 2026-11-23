# Audyt projektu

Audyt dotyczy pierwotnego kodu z commitu `10c364b` i jego zastąpienia pakietem
`src/stock_direction`. Dawne notebooki zachowano w `notebooks/legacy`, wyniki
w `reports/legacy`, a scalone tabele w `data/legacy`. Archiwum zawiera oryginalne
błędy i nie jest obsługiwanym sposobem uruchamiania projektu. Nie należy
interpretować przeniesionych CSV jako wyników poprawionej implementacji.

## Znalezione błędy i sposób naprawy

| Problem w pierwotnym kodzie | Skutek | Obecne rozwiązanie |
| --- | --- | --- |
| `VotingClassifier(voting='hard')` i późniejsze `predict_proba` | Główny eksperyment nie może prawidłowo wyznaczyć prawdopodobieństw | Głosowanie `soft`, zgodne podczas treningu i oceny |
| Optuna ocenia hard voting, finalny SVM ma inną konfigurację | Optymalizacja innego zachowania niż model końcowy | Opcjonalne strojenie LR/RF/SVM przez klonowanie tej samej pipeline; ensemble ma stałą, jawną konfigurację |
| Strojenie dzieli trening 80/20 bez purgingu | Etykiety treningowe sięgają wewnętrznej walidacji | Wewnętrzny `PurgedTimeSeriesSplit` korzysta z dat zakończenia etykiet |
| RFECV i RFE uruchamiane na całej historii | Wynik testowy wpływa na wybrane cechy | Selekcja `SelectKBest` jest częścią pipeline dopasowywanej wewnątrz każdego folda |
| CUSUM odejmuje średnią z całego szeregu | Przyszłość zmienia historyczne cechy | Średnia expanding przesunięta o jeden dzień |
| IsolationForest i scaler dopasowane na pełnej historii, również z `Target` wśród kolumn | Wyciek informacji o przyszłości i etykietach | Usunięto tę ścieżkę; dostępny jest przyczynowy z-score względem wcześniejszych 20 sesji |
| Brak zmienności w rozgrzewce TBM daje klasę neutralną | Sztuczne etykiety dla nieokreślonych barier | Rozgrzewka, zerowa zmienność i niepełny horyzont są pomijane |
| Purging usuwa stałe 5 wierszy także po filtracji obserwacji | Liczba wierszy przestaje oznaczać rzeczywisty czas trwania etykiety | Każda etykieta ma `Label_End`; porównujemy daty zdarzeń |
| Embargo zależy od długości całej, także przyszłej historii i usuwa początek testu | Nowe dane zmieniają stare eksperymenty i kalendarz oceny | Opcjonalny `gap` usuwa dodatkowe obserwacje treningowe; test zachowuje cały dostępny rok |
| `predict_proba[:, 1]` bez mapowania klas; ROC-AUC nie obsługuje brakujących klas | Błędna interpretacja klas lub wyjątek | Kolumny wyrównane przez `classes_`; AUC jest puste, jeśli w teście brakuje wymaganej klasy |
| SVM/RF/XGB w notebookach liczą ROC-AUC z `y_pred` | Raportowane AUC nie mierzy rankingu prawdopodobieństw | Wszystkie modele korzystają ze wspólnej funkcji metryk i prawdopodobieństw |
| Ostatni wiersz binarnego celu bywa oznaczany jako spadek | Fałszywa etykieta bez przyszłej ceny | Jawne wykluczenie nieznanej ostatniej etykiety |
| LR ignoruje zwracany wynik `create_target` | Trening może używać starego `Target` z CSV | Etykiety zawsze tworzone od nowa z surowych cen |
| XGBoost wymaga kolejnych klas, a TBM ma -1/0/1 | Samodzielny XGBoost nie obsługuje etykiet TBM | Adapter koduje klasy do treningu i odtwarza ich znaczenie |
| TBM remapowane na 0/1/2, a AlwaysBuy przewiduje 1 | Baseline przewiduje neutralność zamiast zysku | Jedno kodowanie -1/0/+1 w całym pakiecie; AlwaysUp zawsze +1 |
| `get_sample_weights` wyliczane, lecz nieprzekazywane do treningu | Pozorna obsługa wag; końcowe próbki otrzymują zera | Usunięto martwą ścieżkę; obecna implementacja nie deklaruje ważenia unikalnością |
| Zapisane `n_train`/`n_test` sprzed purgingu i embargo | Raport nie odpowiada liczbie ocenionych próbek | Raportowane są faktyczne liczności i liczba usuniętych próbek |
| `Signal[t] * Log_Return[t]` w wizualizacji | Prognoza następnego dnia zarabia na znanym już zwrocie | Usunięto tę ścieżkę z aktywnego projektu; zapisujemy osobne `Forward_Log_Return`, bez deklarowania wyniku strategii |
| TBM czasem traktowane jak codzienny sygnał -1/0/1, czasem 0/1/2 | Niespójne pozycje i niezgodność z wielodniowymi barierami | Raportujemy klasyfikację; rzeczywisty backtest TBM pozostaje osobnym zadaniem |
| Baseline'y i ensemble mają różne cele oraz okresy oceny | Porównanie jakości jest niemiarodajne | Wspólny cel, cechy, daty i procedura walk-forward dla wskazanych modeli |
| Zależności, ścieżki i seed Optuny nie są ustalone | Trudne odtworzenie eksperymentów | `pyproject.toml`, lockfile, CLI, metadane uruchomienia, hash danych i ustawiany seed |
| Import `pyspark`, ścieżki `../data` z katalogu głównego, puste `except`, globalne wyciszanie ostrzeżeń | Błędy uruchamiania ukrywane w notebookach | Pakiet z małymi modułami, jawne wyjątki, testy i CI |

## Ograniczenia danych i interpretacji

1. **Makroekonomia:** FRED `DATE` oznacza okres obserwacji, a nie moment, od którego
   informacja była dostępna. Sam `ffill` i jednodniowe przesunięcie nie naprawiają
   miesięcznych publikacji i późniejszych rewizji. Archiwalne pliki nie mają
   historii publikacji/vintages, więc nowy eksperyment ich nie używa.
   `merge_available_data` obsługuje złączenie po `Available_At`, ale wymaga
   prawidłowo przygotowanej historii wartości dostępnych w czasie.
   Źródło: [ALFRED — vintage dates i pierwsze publikacje](https://alfred.stlouisfed.org/help/downloaddata).
2. **Sentyment:** README deklarowało FinBERT/VADER, lecz wersjonowane scalone dane
   nie zawierają wyników tych modeli; `headlines.csv` zawiera też rekordy z datą
   1970-01-01. Nie ma podstaw do przypisywania tym eksperymentom korzyści z NLP.
   Aktualizacja: dodano osobną ścieżkę historycznych nagłówków/VADER, walidację pokrycia, duplikatów
   i konserwatywne przypisanie newsów do sesji. Szczegóły oraz ograniczenia
   historycznej dostępności są w [NEWS.md](NEWS.md). Nie zmienia to oceny starych wyników.
3. **Przesuwanie cen:** dawny loader przesuwał wszystkie kolumny oprócz daty,
   włącznie z OHLC. Późniejsze etykietowanie na tych kolumnach nie odpowiadało
   deklarowanemu kalendarzowi. Nowy kod odczytuje bezpośrednio snapshoty cen.
4. **Filtrowanie binarne:** wykluczanie małych *przyszłych* ruchów nie jest regułą
   wyboru transakcji dostępną w chwili decyzji. Opcja `--barrier-std` istnieje do
   eksperymentów warunkowych; domyślnie jest wyłączona, co zmienia zadanie względem
   dawnych wyników. Nie porównuj tych wartości bezpośrednio.
5. **TBM:** implementacja obserwuje zamknięcia, nie bariery intraday. Nie rozstrzyga
   sytuacji, w której High i Low naruszają dwie bariery w jednej sesji.
6. **Interpretacja ekonomiczna:** informacje z zamknięcia sesji nie pozwalają
   automatycznie zawrzeć transakcji po tej samej cenie zamknięcia. Zapisany zwrot
   forward służy do kontroli wyrównania; nie zawiera spreadu, prowizji, opóźnienia
   realizacji ani zasad zarządzania kapitałem.
7. **Dane cenowe:** snapshoty są historycznymi, skorygowanymi danymi dostawcy,
   bez pełnej informacji o pobraniu i korektach. Nowe raporty zapisują ich hash.
   Archiwalny okres został już wykorzystany do eksperymentów, więc nie należy
   przedstawiać go jako wcześniej nietkniętego holdoutu.
8. **Bootstrap:** przedział accuracy nie dowodzi przewagi nad benchmarkiem.
   CLI dodatkowo zapisuje sparowany przedział różnicy względem majority, jeśli
   ten baseline występuje w uruchomieniu. Wybór długości bloków i wielokrotne
   próby modeli nadal wpływają na wnioskowanie. Przy filtracji/TBM bloki kolejnych
   obserwacji nie odpowiadają stałej liczbie sesji.
9. **Kalibracja:** soft voting zakłada użyteczne prawdopodobieństwa. SVM korzysta
   z `CalibratedClassifierCV` z chronologicznymi podziałami i odstępem równym
   maksymalnemu horyzontowi etykiety. Skalowanie i selekcja są wewnątrz
   kalibrowanego estymatora. Raportowana lista cech SVM/ensemble jest sumą cech
   użytych przez składowe kalibracji/modelu, dlatego może być większa niż K.
   Kalibrację nadal trzeba ocenić np. przez Brier score i reliability diagram. Zobacz
   [dokumentację VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html).

## Proponowany kierunek projektu

1. Ustal jedno główne pytanie: czy prosta prognoza kierunku na następną sesję
   poprawia wynik względem majority i AlwaysUp na tych samych datach? Ensemble
   uzasadniaj dopiero po porównaniu z LR i RF oraz kosztu obliczeń.
2. Zamroź protokół i zakres strojenia przed kolejnymi badaniami. Raportuj wyniki
   osobno dla spółek, lat i reżimów zmienności; dodaj ablation study grup cech.
   Zarezerwuj nowy okres na końcową ocenę, bez dobierania cech na jego wynikach.
3. Dodaj rzeczywisty backtest z wejściem najwcześniej po utworzeniu sygnału,
   kosztami, spreadem, poślizgiem i zdefiniowaną ekspozycją. Dla TBM obsłuż
   rzeczywiste wyjścia barierowe, nakładające się pozycje i alokację kapitału.
   Raportuj obrót, drawdown i przewagę nad buy-and-hold na pełnym kalendarzu.
4. Dopiero potem włącz dane makro z datami publikacji i wersjami ALFRED, a newsy
   po audycie timestampów. Sprawdź, czy dodają wartość ponad same ceny.
5. Rozszerz uniwersum poza trzy wybrane obecnie duże spółki technologiczne,
   kontrolując survivorship bias, korelację spółek i zmiany składu uniwersum.
6. Porównuj modele sparowanym bootstrapem różnic oraz kontroluj wielokrotne
   testowanie; nie wybieraj „najlepszego” wyłącznie po jednym zagregowanym accuracy.

## Zmiana struktury

Kod użytkowy jest w `src/stock_direction`; eksperymenty w `notebooks`;
automatyczne testy w `tests`; dokumentacja w `docs`; dane wejściowe w `data/raw`;
raporty nowych uruchomień w ignorowanym `reports/runs`. Surowe dane, historyczne
wyniki i prezentacja zostały przeniesione bez zmiany zawartości. Stare ścieżki
w notebookach archiwalnych odnoszą się do pierwotnego układu repozytorium.
Użytkownik korzystający z nowych ścieżek powinien uruchamiać CLI lub nowy notebook.

Dokumentacja metody podziału:
[TimeSeriesSplit i parametr gap](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
Sam `gap` liczony w wierszach nie zastępuje kontroli `Label_End` po filtracji.
