# Trappist Tour

Projekt për lëndën "Dizajni dhe Analiza e Algoritmeve", bazuar në sfidën
SpOC: Trappist Tour të Agjencisë Evropiane të Hapësirës (ESA).

## Problemi

Një sondë hapësinore niset nga një largësi prej 10 AU rreth yllit TRAPPIST-1
dhe duhet t'i vizitojë të shtatë planetët e sistemit (b, c, d, e, f, g, h),
secilin saktësisht një herë. Rendi i vizitës është i lirë.

Në çdo planet sonda mund të bëjë një flyby (asistencë gravitacionale, pa kosto
karburanti). Në çdo kalim ndërmjet dy planetëve lejohet maksimumi një manovër
në hapësirë të thellë (DSM), që harxhon karburant.

Kufizimi: sonda nuk guxon të afrohet te ylli më pak se 834840 km.

## Objektivat

Problemi është multi-objektiv, me dy objektiva që bien ndesh:

1. Minimizimi i ΔV (karburantit të harxhuar).
2. Minimizimi i kohës totale të misionit (T).

Zgjidhja nuk është një rrugë e vetme, por një bashkësi zgjidhjesh që përafron
Pareto frontin. Cilësia matet me hypervolume, ku llogariten vetëm zgjidhjet
me ΔV < 2500 m/s dhe T < 4000 ditë.

## Qasja

1. Baseline: random search.
2. NSGA-II (pygmo).
3. Algoritëm i dytë për krahasim.
4. NSGA-II i ndihmuar nga model zëvendësues (surrogate), i trajnuar në GPU.
5. Krahasim me hypervolume dhe analizë e kompleksitetit.

## Struktura e repo-s

- src/ : kodi burimor
- notebooks/ : notebooks (Kaggle)
- results/ : rezultatet e eksperimenteve
- report/ : raporti final

## Si ekzekutohet

Do te plotesohet gjate punes.

## Rezultatet

Do te plotsohen pas eksperimenteve.