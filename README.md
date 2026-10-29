# Photonic entanglement swapping

Code and data for *A mode-mixing threshold for entanglement swapping with one independent auxiliary photon*, by Annan Chen.

## Run

Use Python 3.12 and install the dependencies:

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

To regenerate the datasets from the source states and optical networks:

```sh
python reproduce.py data
python reproduce.py
```

The commands write numerical results to `results/` and Figures 1–4 to `figures/` in PDF, SVG and PNG formats. The repository includes the four PDF figures.

## Files

- `src/`: physical models and acceptance policies.
- `scripts/`: data, result and figure generation.
- `data/`: the datasets used in the article.
- [DATA_DICTIONARY.md](DATA_DICTIONARY.md): parameters, array fields and probability conventions.

## Cite and reuse

Citation metadata are in [CITATION.cff](CITATION.cff).

Code and documentation use the [MIT license](LICENSE). Data and scientific figures use [CC BY 4.0](LICENSE-DATA).
