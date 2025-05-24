# **Projekt IUM etap 1**
-  **_Michał Patasz_**
-  **_Michał Sadlej_**


## **Definicja problemu biznesowego**
Potrzeba automatyzacji i usprawnienia procesu wypełniania pól podczas dodawania
nowych ofert w celu zwiększenia efektywności, zmniejszenia liczby błędów oraz
poprawy doświadczenia użytkownika.


## **Zadania modelowania**
1. **Automatyczne uzupełnianie danych podstawowych**
   - Przewidywanie wartości pól na podstawie danych użytkownika (np. adres)

2. **Klasyfikacja/kategoryzacja ofert**
   - Automatyczne przypisywanie oferty do kategorii (np. apartament, hotel, domek letniskowy) na podstawie opisu i zdjęć
   - *Uwaga: Wymaga dostępu do listings.csv ze szczegółami ofert*

3. **Sugerowanie wartości dla pól opisowych**
   - Generowanie rekomendacji dla pól tekstowych na podstawie podobnych, wcześniej utworzonych ofert i pozytywnych recenzji.

4. **Wykrywanie potencjalnych błędów**
   - Identyfikacja niespójności lub brakujących danych podczas wypełniania formularza.


## **Założenia projektu**
1. Wzorce wypełniania formularzy przez użytkowników wykazują pewną
powtarzalność i strukturę możliwą do wychwycenia przez algorytmy uczenia
maszynowego.
2. Dane historyczne z recenzji zawierają informacje pozwalające określić, które
parametry oferty przyczyniają się do jej sukcesu.
3. Użytkownicy będą akceptować sugestie systemu, jeśli będą one trafne i
oszczędzające czas.
4. Automatyzacja nie musi być kompletna - częściowe wypełnienie formularza
również przyniesie wartość biznesową.
5. System musi być intuicyjny i przyjazny dla oferentów, którzy mogą nie mieć
doświadczenia w dodawaniu ofert, a jednocześnie powinien umożliwiać ręczne
nadpisywanie automatycznych sugestii, aby zachować pełną kontrolę dla
użytkowników nieufnych wobec automatyzacji.
6. System nie może pogorszyć jakości danych.


## **Proponowane kryteria sukcesu**

### **Kryteria biznesowe:**
- Łączny czas przeznaczony na dodanie oferty do systemu.
- Zwiększenie liczby nowych ofert - zauważalny wzrost po wdrożeniu rozwiązania.
- Redukcja liczby negatywnych opinii spowodowanych nieporozumieniami między
klientem a właścicielem obiektu.

### **Kryteria analityczne:**
- Dokładność predykcji - określa odsetek pól pozostawionych bez zmian przez
użytkownika po ich automatycznym wypełnieniu przez algorytm - ponad 50%.
- Pokrycie formularza - procent pól, dla których system jest w stanie
zaproponować wartości - ponad 50%.
- Czas odpowiedzi - czas potrzebny na wygenerowanie sugestii - mniej niż 1s.

### **Weryfikacja baseline’u:**
- Porównanie z modelem naiwnym (np. zawsze sugeruj "Kraków" dla lokalizacji).
- Do dokładniejszych badań potrzebne będą dane z listings.csv.


## **Analiza danych**

### **Users.csv:**
1. Zakładamy, że ‘Id’ użytkownika powinno być unikalne.
2. Rekordy z pustym polem ‘Id’ użytkownika mogą być odrzucone. W tej chwili
odrzuconych pól byłoby ponad 12 tysięcy (około 20% dostępnych danych).
3. Dane adresowe powinny być poprawne oraz kompletne.

### **Reviews.csv:**
1. ‘Listing_id’ powinno odpowiadać odpowiedniemu ‘Id’ oferty w pliku ‘listings.csv’.
2. Data i dane autora powinny być poprawne.
3. Pola ‘comment’ nie powinny być puste. Puste komentarze mogą być odrzucone.
4. W obecnych danych rekordy, które nie posiadają ‘listing_id’ lub ‘comment’to
około 26 tysięcy (35% dostępnych danych).

### **Listings.csv:**
1. Powinniśmy posiadać dostęp do pliku z ofertami.
2. W tym pliku powinny znajdować się wartości pól z szczegółami ofert oraz
unikalnym id oferty.


### **Rozkłady kluczowych atrybutów**

- Najszęściej występujące miasta:

- Rozkład liczby recenzji dla ofert:

- Rozkład długości recenzji:
