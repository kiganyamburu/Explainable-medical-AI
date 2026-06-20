import os
from app import create_app

app = create_app()

if __name__ == '__main__':
    # Fetch port configuration from environment variables (useful for Render/Railway)
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "t")
    
    print(f"Starting server in debug={debug} on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)
