release: git lfs pull && pip install -r requirements.txt
web: gunicorn --bind 0.0.0.0:$PORT app:app
