.PHONY: setup chunks index ui
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
chunks:
	. .venv/bin/activate && python scripts/01_build_chunks.py
index:
	. .venv/bin/activate && python scripts/02_embed_and_index_mongo.py
ui:
	. .venv/bin/activate && streamlit run app/ui_streamlit.py
