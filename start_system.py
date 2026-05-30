#!/bin/bash
"""
Startup script for Medical Evidence Graph & Outcomes Insight Lab
Launches both API backend and frontend interface
"""

import os
import sys
import subprocess
import threading
import time
import signal
import requests
from pathlib import Path


def check_port(port):
    """Check if port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0


def start_api_server():
    """Start the FastAPI backend server"""
    print("🚀 Starting API backend server on port 8000...")
    
    try:
        # Start the API server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.api_backend:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"  # Enable auto-reload during development
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if the server started successfully
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API backend server started successfully!")
                return process
            else:
                print(f"❌ API server returned unexpected status: {response.status_code}")
                return None
        except requests.exceptions.RequestException:
            print("❌ Failed to connect to API server")
            return None
    
    except Exception as e:
        print(f"❌ Error starting API server: {str(e)}")
        return None


def start_frontend_interface():
    """Start the Streamlit frontend interface"""
    print("🎨 Starting frontend interface on port 8501...")
    
    try:
        # Start the Streamlit app
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "src.frontend_interface:main",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if the server started successfully
        try:
            response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
            if response.status_code == 200:
                print("✅ Frontend interface started successfully!")
                return process
            else:
                print(f"❌ Frontend server returned unexpected status: {response.status_code}")
                return None
        except requests.exceptions.RequestException:
            print("⚠️  Could not verify frontend startup (this is normal during initial load)")
            return process
    
    except Exception as e:
        print(f"❌ Error starting frontend interface: {str(e)}")
        return None


def main():
    """Main startup function"""
    print("=" * 60)
    print("🏥 MEDICAL EVIDENCE GRAPH & OUTCOMES INSIGHT LAB")
    print("Phase 3: Clinical Decision Support System Startup")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        "src/api_backend.py",
        "src/frontend_interface.py"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Required file not found: {file}")
            return 1
    
    print("✅ All required files exist")
    
    # Check if ports are available
    if not check_port(8000):
        print("❌ Port 8000 (API) is already in use")
        return 1
        
    if not check_port(8501):
        print("❌ Port 8501 (Frontend) is already in use")
        return 1
    
    print("✅ Ports are available")
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        print("❌ Failed to start API server, stopping.")
        return 1
    
    # Wait a bit more for API to fully initialize
    time.sleep(2)
    
    # Start frontend interface
    frontend_process = start_frontend_interface()
    if not frontend_process:
        print("⚠️  Frontend interface failed to start, but API is running")
        print("💡 You can still use the API directly at http://localhost:8000")
        print("🔧 Or try starting the frontend separately with: streamlit run src/frontend_interface.py")
    else:
        print("✅ Both API and frontend started successfully!")
    
    print("\n" + "=" * 60)
    print("SYSTEM NOW RUNNING:")
    print("🌐 API Backend: http://localhost:8000/docs")
    print("🖥️  Frontend Interface: http://localhost:8501")
    print("📋 Health Check: http://localhost:8000/health")
    print("=" * 60)
    print("\n💡 Press Ctrl+C to shutdown all services\n")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
            # Check if processes are still running
            if api_process and api_process.poll() is not None:
                print("❌ API server stopped unexpectedly")
                break
            if frontend_process and frontend_process.poll() is not None:
                print("⚠️  Frontend interface stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        
        # Terminate processes
        if api_process:
            api_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        
        print("✅ Shutdown complete")
        return 0


if __name__ == "__main__":
    sys.exit(main())