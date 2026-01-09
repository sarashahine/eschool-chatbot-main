from app_arabic import create_app

from config import ARABIC_PORT, FLASK_DEBUG

arabic_app = create_app()

if __name__ == "__main__":
    debug = FLASK_DEBUG.lower() == "true"
    arabic_app.run(port=ARABIC_PORT, debug=debug)
