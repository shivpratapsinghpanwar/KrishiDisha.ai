"""WSGI entry point.

    python app.py                 # development server
    flask --app app run --debug   # alternative
    gunicorn app:app              # production (see Procfile)

All behaviour lives in the ``krishidisha`` package; this file only builds the app.
"""
import os

from krishidisha import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1") not in ("0", "false", "False")
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=port, debug=debug)
