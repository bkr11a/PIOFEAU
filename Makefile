.PHONY: compile_latex move_pdf journal_article clean_latex clean requirements_file activate_venv start_tensorboard_session end_tensorboard_session build_docs

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
BIBLIOGRAPHY_NAME = phys
PDF = $(ARTICLE_NAME).pdf
LATEX = $(ARTICLE_NAME).tex
BIB_FILE = $(BIBLIOGRAPHY_NAME).bib
TEX_DIR = reports/typesetting
ARTICLE_DIR = reports

VENV_DIR = POIFE-Unrolling-venv

###############
#  COMMANDS   #
###############

# Clean ALL Directories
clean:
	rm -rf assets;
	rm -rf data;
	rm -rf docs;
	rm -rf notebooks;
	rm -rf $(VENV_DIR);
	rm -rf references;
	rm -rf reports;
	rm -rf src;
	rm -rf tests;
	rm -rf .env .gitignore cleanup.sh LICENSE Makefile poetry.lock pyproject.toml README.md requirements.txt setup.cfg

# Push requirements out to a requirements.txt
requirements_file:
	pip freeze > requirements.txt

# LATEX Journal Article Commands #
journal_article: compile_latex move_pdf clean_latex

compile_latex:
	cd $(TEX_DIR) && pdflatex "$(LATEX)" && bibtex "$(ARTICLE_NAME)" && pdflatex "$(LATEX)" && pdflatex "$(LATEX)"

move_pdf:
	mv "$(TEX_DIR)/$(PDF)" "$(ARTICLE_DIR)/$(PDF)"

clean_latex:
	cd $(TEX_DIR) && rm -f "$(ARTICLE_NAME).aux" "$(ARTICLE_NAME).toc" "$(ARTICLE_NAME).out" "texput.fls" "$(ARTICLE_NAME).bbl" "$(ARTICLE_NAME).blg"

activate_venv:
	pwd && source ${VENV_DIR}/bin/activate

start_tensorboard_session:
	pwd && ls -la && source runTensorboardSession.sh

end_tensorboard_session:
	source killTensorboardSession.sh

build_docs:
	cd docs && mkdocs build