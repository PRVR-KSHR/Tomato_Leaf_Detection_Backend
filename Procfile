release: git lfs pull && pip install -r requirements.txt
web: gunicorn --workers=1 --threads=2 --worker-class=sync --max-requests=100 --bind 0.0.0.0:$PORT app:app
