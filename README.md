# Predikcija Cijena Kuća
Ovaj projekt koristi duboku neuronsku mrežu za predikciju cijena kuća.


## Izvor podataka
Podaci korišteni u ovom projektu preuzeti su s Kagglea: House Prices Dataset. 

CSV datoteka sample_submission.csv preimenovana je u house-price.csv radi bolje organizacije. 

Iz house-price.csv datoteke uklonjen je stupac id, a ostao je samo stupac SalePrice.

[House Price Dataset](https://www.kaggle.com/datasets/lespin/house-prices-dataset)


## Potrebne biblioteke
Za pokretanje projekta, potrebno je instalirati sljedeće biblioteke:

- Pandas
- Torch
- Numpy
- Matplotlib
- Scikit-learn

Ako ih nemate instalirane, možete ih instalirati pomoću sljedeće naredbe:

```bash
pip install pandas torch numpy matplotlib scikit-learn
```

## Kako pokrenuti projekt
Klonirajte repozitorij ili preuzmite projekt na svoj lokalni uređaj.

Provjerite da li se CSV datoteka house-price.csv nalazi u istom direktoriju kao i Python skripta.

Pokrenite Python skriptu da biste izvršili model predikcije.