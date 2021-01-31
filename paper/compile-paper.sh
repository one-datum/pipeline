#!/usr/bin/env bash

pdflatex one-datum.tex
bibtex one-datum
pdflatex one-datum.tex
pdflatex one-datum.tex
