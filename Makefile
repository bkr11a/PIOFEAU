.PHONY: compile_latex move_pdf journal_article clean_latex

###############
#   GLOBALS   #
###############

PROJECT_NAME = PIOFE-Unrolling
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

###############
#  VARIABLES  #
###############

ARTICLE_NAME = Physics-Informed Optical Flow Estimation via Algorithm Unrolling
PDF = $(ARTICLE_NAME).pdf
LATEX = $(ARTICLE_NAME).tex
TEX_DIR = reports/typesetting
ARTICLE_DIR = reports

###############
#  COMMANDS   #
###############

journal_article: compile_latex move_pdf clean_latex

compile_latex:
	cd $(TEX_DIR) && pdflatex "$(LATEX)"

move_pdf:
	mv "$(TEX_DIR)/$(PDF)" "$(ARTICLE_DIR)/$(PDF)"

clean_latex:
	cd $(TEX_DIR) && rm -f "$(ARTICLE_NAME).aux" "$(ARTICLE_NAME).toc" "$(ARTICLE_NAME).out" "texput.fls"