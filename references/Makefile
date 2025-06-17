.PHONY : yamls

# Pdfs not starting with an underscore to be processed.
PDFS = $(filter-out _%.pdf,$(wildcard *.pdf))
YAMLS = ${PDFS:.pdf=.yaml}

yamls : ${YAMLS}

${YAMLS} : %.yaml : %.pdf ../src/hiv/scripts/ai_review.py
	python -m hiv.scripts.ai_review $< $@
