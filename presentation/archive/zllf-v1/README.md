# Kelpod pitch

Status-update pitch (6–7 minutes, one speaker) for Frontier Systems 2026/27. Same format as the skill-following talk: blue Beamer (Madrid-style) frame, one claim per slide, click-by-click animations, full speaker script.

- `index.html`: the deck. Open it in a browser. Arrow keys or click to step, `N` for speaker notes, `F` for fullscreen.
- `pdf/zllf-pitch-slides.pdf`: one page per slide (final state). Send this one to the course organisers for the shared deck.
- `pdf/zllf-pitch-frames.pdf`: one page per click step.

Re-export the PDFs after editing:

```bash
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for m in slides frames; do
  "$CH" --headless=new --no-pdf-header-footer --virtual-time-budget=25000 \
    --print-to-pdf="$PWD/pdf/zllf-pitch-$m.pdf" "file://$PWD/index.html?print=$m"
done
```

Fonts and the animation library load from the web, so present from the PDF if the room has no internet.
