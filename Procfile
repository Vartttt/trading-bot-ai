echo 'web: gunicorn app.app:app --bind 0.0.0.0:${PORT}' > Procfile
grep -q '^gunicorn' requirements.txt || echo 'gunicorn==21.2.0' >> requirements.txt

