# Pravila unosa turističkih činjenica

## Obavezna polja

Svaki red u `data/raw/facts.jsonl` mora biti jedan validan JSON objekat sa poljima:

- `fact_id`
- `destination_id`
- `category`
- `text`
- `source_id`
- `is_dynamic`
- `last_verified_at`

Polje `valid_until` koristi se samo kada postoji poznat datum do kojeg je dinamička informacija važeća.

## Dozvoljene kategorije

- `description`
- `attraction`
- `history`
- `nature`
- `activity`
- `food`
- `practical`
- `route`

## Minimalna pokrivenost destinacije

Svaka destinacija mora imati najmanje šest činjenica:

1. jedan osnovni opis;
2. dvije atrakcije ili znamenitosti;
3. jednu aktivnost;
4. jednu preporuku za sezonu ili tip posjetioca;
5. jednu vezu sa obližnjom destinacijom.

## Pravila sadržaja

- Jedan red sadrži jednu jasnu tvrdnju ili preporuku.
- Tekst se piše svojim riječima; ne kopiraju se cijeli pasusi iz izvora.
- Jezik je srpski, ijekavica, latinica.
- Ne koriste se promotivni superlativi koji nisu podržani izvorom.
- Svaki `source_id` mora postojati u `data/sources.csv`.
- Ako izvor ne podržava tvrdnju dovoljno jasno, činjenica se ne unosi.
- Istorijske i geografske tvrdnje iz sekundarnog izvora treba potvrditi drugim pouzdanim izvorom kada je moguće.

## Dinamičke informacije

`is_dynamic` je `true` samo za podatke koji se mogu promijeniti, kao što su:

- cijene;
- radno vrijeme;
- kontakt;
- red vožnje;
- datum događaja.

Dinamička činjenica mora imati `valid_until` ili u tekstu jasnu napomenu da informaciju treba provjeriti prije putovanja.

## Konvencije ID vrijednosti

Primjer:

- `F-SARAJEVO-001`
- `F-JAJCE-001`
- `F-MARTIN-BROD-NP-UNA-001`

Brojač počinje od `001` zasebno za svaku destinaciju.

## Dužina teksta

Radi usklađivanja sa kasnijom validacijom, tekst treba imati između 30 i 500 znakova.
