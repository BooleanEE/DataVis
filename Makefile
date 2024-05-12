save:
	pip freeze > requirements.txt

install:
	pip install -r requirements.txt

run:
	streamlit run main.py