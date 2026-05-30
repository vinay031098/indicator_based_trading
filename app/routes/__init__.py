"""Route blueprints."""

from app.routes.analysis import analysis_bp
from app.routes.auth_routes import auth_bp
from app.routes.chat import chat_bp
from app.routes.data import data_bp
from app.routes.meta import meta_bp

ALL_BLUEPRINTS = [meta_bp, auth_bp, analysis_bp, data_bp, chat_bp]
