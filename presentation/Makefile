# Kelpod slides. Needs Tectonic (brew install tectonic); it downloads TeX packages on first use.
TECTONIC ?= tectonic

.PHONY: slides handout notes clean

slides:
	mkdir -p build pdf
	$(TECTONIC) --outdir build main.tex
	cp build/main.pdf pdf/kelpod-slides.pdf

handout:
	mkdir -p build pdf
	$(TECTONIC) --outdir build handout.tex
	cp build/handout.pdf pdf/kelpod-handout.pdf

notes:
	mkdir -p build pdf
	$(TECTONIC) --outdir build notes.tex
	cp build/notes.pdf pdf/kelpod-notes.pdf

clean:
	rm -rf build
