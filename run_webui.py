#!/usr/bin/env python3
"""
Launcher script for Supernova Web UI
Simple wrapper to run the Streamlit application
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Supernova Web UI"""
    
    # Get the current directory
    current_dir = Path(__file__).parent
    web_ui_path = current_dir / "web_ui.py"
    
    # Check if web_ui.py exists
    if not web_ui_path.exists():
        print("❌ Error: web_ui.py not found!")
        print("Make sure you're running this from the USLM directory.")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found!")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    print("🌟 Starting Supernova Web UI...")
    print("📱 The web interface will open in your browser automatically")
    print("🔗 If it doesn't open, navigate to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Run streamlit
    try:
        # Change to the directory containing web_ui.py
        os.chdir(current_dir)
        
        # Run streamlit with the web UI
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(web_ui_path),
            "--server.headless", "false",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--theme.base", "dark"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Supernova Web UI...")
        print("👋 Thanks for using Supernova!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
