#!/usr/bin/env python3
"""
Enhanced Launcher for Supernova Web UI
Provides choice between basic and enhanced (with tools) interfaces
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Enhanced Supernova Web UI"""
    
    # Get the current directory
    current_dir = Path(__file__).parent
    enhanced_ui_path = current_dir / "enhanced_web_ui.py"
    
    # Check if enhanced_web_ui.py exists
    if not enhanced_ui_path.exists():
        print("❌ Error: enhanced_web_ui.py not found!")
        print("Make sure you're running this from the USLM directory.")
        sys.exit(1)
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found!")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    print("🌟 Starting Enhanced Supernova Web UI...")
    print("🛠️  This version includes advanced tools:")
    print("   🧮 Math solving with SymPy + Wolfram Alpha")
    print("   💻 Python code execution with safety checks")  
    print("   📊 Data plotting with Matplotlib")
    print("   🔍 Web search integration")
    print("   🛡️  Safety filtering and content control")
    print("")
    print("📱 The enhanced web interface will open in your browser")
    print("🔗 If it doesn't open, navigate to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    # Run streamlit
    try:
        # Change to the directory containing enhanced_web_ui.py
        os.chdir(current_dir)
        
        # Run streamlit with the enhanced web UI
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(enhanced_ui_path),
            "--server.headless", "false",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--theme.base", "dark"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Enhanced Supernova Web UI...")
        print("👋 Thanks for using Supernova Enhanced!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Enhanced Streamlit: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
