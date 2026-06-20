from flask import Blueprint, render_template

web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    """Serves the single-page application landing and dashboard UI."""
    return render_template('index.html')
