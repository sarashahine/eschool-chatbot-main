from app_nomic import create_app

from config import NOMIC_PORT, FLASK_DEBUG

nomic_app = create_app()

if __name__ == "__main__":
    debug = FLASK_DEBUG.lower() == "true"
    nomic_app.run(port=NOMIC_PORT, debug=debug)
