# T14 — Evaluation rubric

Ovaj dokument definiše kako se tumače `expected_points` u
`data/evaluation/evaluation_questions.jsonl`.

## Osnovno pravilo

Evaluacioni skup ima tačno 60 pitanja/scenarija i koristi se identično prije i poslije fine-tuninga.
`expected_points` nisu jedan obavezni referentni odgovor. Dobar odgovor može biti drugačije formulisan,
ali treba sadržati ključne informacije i ponašanja navedena za dati primjer.

## Kategorije

- **destination (15)** — odgovor treba dati tačne, relevantne i praktično organizovane informacije o konkretnoj destinaciji.
- **lesser_known (10)** — odgovor treba kvalitetno predstaviti manje poznato mjesto bez izmišljanja sadržaja samo zato što je destinacija manje poznata.
- **trip_plan (10)** — plan treba imati realan redoslijed, ograničen broj glavnih tačaka i uvažavati broj dana/prevoz iz pitanja.
- **personalized (8)** — preporuka mora jasno koristiti navedene interese, kondiciju, društvo ili stil putovanja.
- **multi_turn (5)** — `messages` sadrži više uzastopnih korisničkih poruka. Evaluacioni runner treba slati korisničke poruke redom i između njih sačuvati odgovor modela. Završni odgovor mora koristiti informacije iz prethodnih korisničkih poruka i ne smije ponovo tražiti podatke koje je korisnik već dao.
- **dynamic (5)** — cijene, radno vrijeme, redovi vožnje, dostupnost i slični podaci ne smiju se izmišljati. Model treba jasno reći da informaciju treba provjeriti iz aktuelnog izvora.
- **out_of_domain (4)** — odgovor treba kratko zadržati ulogu turističkog vodiča za BiH i preusmjeriti korisnika na turističku temu.
- **ambiguous (3)** — odgovor treba tražiti potrebnu dopunu umjesto da sam izmisli destinaciju, broj dana, prevoz ili interesovanja.

## `expected_points`

Za svako pitanje:
- ne zahtijeva se ista formulacija;
- svaki očekivani element treba prepoznati semantički;
- odgovor ne mora navesti sve moguće turističke činjenice;
- važniji su tačnost, relevantnost i korisnost nego dužina.

## `must_not_include`

Ovo polje navodi tipične greške koje odgovor ne smije sadržati, na primjer:
- izmišljenu cijenu;
- izmišljeno radno vrijeme ili red vožnje;
- ignorisanje prethodnog multi-turn konteksta;
- detaljan odgovor na pitanje koje je van turističkog domena;
- samouvjerenu pretpostavku kada je pitanje nejasno.

## Stabilnost skupa

Nakon što se T14 commit završi i prije baseline evaluacije provjeri se:
1. tačno 60 pitanja/scenarija;
2. raspodjela 15 / 10 / 10 / 8 / 5 / 5 / 4 / 3;
3. svaki primjer ima `question_id`, `category`, `messages` i `expected_points`;
4. nijedan `user` tekst nije identičan user poruci iz finalnog SFT skupa;
5. `question_id` i user tekstovi se ne ponavljaju unutar evaluacionog skupa.

Nakon baseline testa ovaj evaluacioni skup se ne mijenja.
