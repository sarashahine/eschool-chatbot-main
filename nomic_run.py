from app import create_app

from config import NOMIC_PORT, FLASK_DEBUG

nomic_app = create_app()

if __name__ == "__main__":
    debug = FLASK_DEBUG.lower() == "true"
    nomic_app.run(host="0.0.0.0", port=NOMIC_PORT, debug=debug)
