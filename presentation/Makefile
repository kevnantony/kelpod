# Kelpod slides. Needs Tectonic (brew install tectonic); it downloads TeX packages on first use.
TECTONIC ?= tectonic

.PHONY: slides notes clean

slides:
	mkdir -p build pdf
	$(TECTONIC) --outdir build main.tex
	mkdir -p pdf && cp build/main.pdf pdf/kelpod-slides.pdf

notes:
	mkdir -p build pdf
	$(TECTONIC) --outdir build notes.tex
	mkdir -p pdf && cp build/notes.pdf pdf/kelpod-notes.pdf

clean:
	rm -rf build
