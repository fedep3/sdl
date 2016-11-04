NAME=paper

all : $(NAME).pdf

$(NAME).pdf : $(wildcard *.tex) bibliography.bib
	pdflatex $(NAME).tex
	bibtex $(NAME)
	pdflatex $(NAME).tex
	pdflatex $(NAME).tex

clean : 
	@rm -f *.aux *.bbl *.blg *.dvi *.lof *.log *.lot *.out *.pdf *.pdf_tex *.svg *.toc *~ source.tar.gz


view: $(NAME).pdf
	@xdg-open $(NAME).pdf 1>/dev/null 2>/dev/null

tex: $(NAME).tex
	@xdg-open $(NAME).tex 1>/dev/null 2>/dev/null

tarball: $(NAME).pdf
	@tar cfvz source.tar.gz acm_proc_article-sp.cls IEEEtran.bst *.tex *.bib Makefile figures/
