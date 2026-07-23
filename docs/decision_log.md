# Decision log

## 2026-07-23 — Uklanjanje Colab-specifične lokalne zavisnosti

**Problem:** `pip freeze` je u `requirements-lock.txt` dodao paket
`google-colab` preko lokalne putanje
`/colabtools/dist/google_colab-1.0.0.tar.gz`.

**Odluka:** Uklonjen je samo red za lokalnu instalaciju paketa
`google-colab`.

**Razlog:** Lokalna Colab putanja ne postoji u novom runtimeu i zbog
toga sprečava instalaciju zaključanih projektnih zavisnosti.

**Posljedica:** `requirements-lock.txt` sadrži prenosive verzije
projektnih paketa, dok Colab i dalje automatski obezbjeđuje svoj
sistemski paket.

