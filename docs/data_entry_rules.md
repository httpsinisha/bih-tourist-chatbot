# Pravila unosa turističkih činjenica

## Namjena

Datoteka `data/raw/facts.jsonl` je strukturisana zbirka provjerenih i
preformulisanih turističkih činjenica. Svaki red je jedan samostalan JSON
objekat i jedna jasna tvrdnja ili preporuka.

## Obavezna polja

Svaki objekat mora sadržati:

- `fact_id`
- `destination_id`
- `category`
- `text`
- `source_id`
- `is_dynamic`
- `last_verified_at`

Polje `valid_until` dodaje se samo kada postoji poznat datum do kojeg važi
dinamička informacija.

## Dozvoljene kategorije

- `description`
- `attraction`
- `history`
- `nature`
- `activity`
- `food`
- `practical`
- `route`

## Minimalna pokrivenost

Svaka od 72 destinacije mora imati najmanje šest činjenica:

1. osnovni opis;
2. dvije turističke vrijednosti, kao što su znamenitosti, istorija ili priroda;
3. jednu aktivnost;
4. jednu preporuku za sezonu ili tip posjetioca;
5. jednu vezu sa obližnjom destinacijom.

Kompletni paket ima tačno šest činjenica po destinaciji, ukupno 432.

## Pravila pisanja

- Tekst se piše na srpskom jeziku, ijekavicom i latinicom.
- Jedan red sadrži samo jednu jasnu tvrdnju ili preporuku.
- Tekst mora biti između 30 i 500 znakova.
- Ne kopiraju se cijeli pasusi sa izvora.
- Ne koriste se marketinški superlativi kao činjenične tvrdnje.
- Svaki `source_id` mora postojati u `data/sources.csv` i imati status
  `approved`.
- Nepotvrđena tvrdnja se ne unosi.
- Za osjetljiva memorijalna mjesta koristi se neutralan i dostojanstven jezik.
- Za planinarenje, rafting i slične aktivnosti navodi se potreba za provjerom
  vremena, opreme, lokalnih pravila ili stručnog vodiča gdje je relevantno.

## Dinamičke informacije

`is_dynamic` je `true` samo za:

- cijene;
- radno vrijeme;
- kontakte;
- red vožnje;
- datume događaja;
- druge podatke koji se često mijenjaju.

Dinamička činjenica mora imati `valid_until` ili jasnu napomenu da se podatak
provjeri prije putovanja. Ovaj T08 paket ne sadrži dinamičke činjenice.

## Konvencija ID vrijednosti

Primjeri:

- `F-SARAJEVO-001`
- `F-JAJCE-001`
- `F-MARTIN-BROD-I-NP-UNA-001`

Brojač počinje od `001` zasebno za svaku destinaciju.

## Datumi

`last_verified_at` se zapisuje u ISO formatu `YYYY-MM-DD`.
