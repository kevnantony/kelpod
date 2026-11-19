# Kelpod slides

Beamer slides for the Kelpod status update (Frontier Systems, WiSe 2026/2027). Madrid theme, in the same blue palette as the earlier decks.

## Files

| File | What it is |
|---|---|
| `main.tex` | The slides, with speaker notes in `\note{...}` |
| `notes.tex` | Wrapper that builds the notes version (`make notes`) |
| `Makefile` | `make slides`, `make notes`, `make clean` |
| `pdf/kelpod-slides.pdf` | Built slides (commit this one for the course) |
| `pdf/kelpod-notes.pdf` | Slides with speaker notes, two per page |
| `archive/zllf-v1/` | The earlier Zero-Leak LoRA Factory deck (HTML), kept for history |

## Build

You need [Tectonic](https://tectonic-typesetting.github.io/) (`brew install tectonic`). The first build downloads the TeX packages it needs, so it takes a while and needs internet access.

```bash
cd presentation
make slides   # -> pdf/kelpod-slides.pdf
make notes    # -> pdf/kelpod-notes.pdf
```

## Editing

- Each slide is one `\begin{frame}...\end{frame}`. Put the speaking notes in `\note{...}` at the end of the frame.
- Colours are defined at the top of `main.tex`.
- Numbers on the slides come from the pivot document (`Kelpod_Final.pdf`). Targets are development targets, not validated results, and the slides say so.
