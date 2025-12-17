from app import create_app

from config import PORT, FLASK_DEBUG

app = create_app()

if __name__ == "__main__":
    debug = FLASK_DEBUG.lower() == "true"
    app.run(host="0.0.0.0", port=PORT, debug=debug)
