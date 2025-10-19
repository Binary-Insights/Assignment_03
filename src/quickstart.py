"""
Quickstart Launcher for MATLAB RAG Assistant
Provides interactive menu to start backend, frontend, or both
Location: quickstart.py
"""

import subprocess
import sys
import time
import webbrowser
import os
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header():
    """Print application header"""
    print("\n" + "=" * 70)
    print(f"{BOLD}{BLUE}MATLAB RAG Assistant - Quickstart Launcher{RESET}")
    print("=" * 70)
    print(f"{BLUE}Frontend{RESET}: Streamlit (port 8501)")
    print(f"{BLUE}Backend{RESET}: FastAPI (port 8000)")
    print("=" * 70 + "\n")


def print_menu():
    """Print main menu"""
    print(f"\n{BOLD}Select an option:{RESET}\n")
    print(f"  {GREEN}1{RESET}) Start FastAPI Backend only")
    print(f"  {GREEN}2{RESET}) Start Streamlit Frontend only")
    print(f"  {GREEN}3{RESET}) Start Both (Backend + Frontend) - RECOMMENDED")
    print(f"  {GREEN}4{RESET}) Check API Health")
    print(f"  {GREEN}5{RESET}) View API Documentation")
    print(f"  {GREEN}6{RESET}) Exit")
    print()


def start_backend():
    """Start FastAPI backend server"""
    print(f"\n{YELLOW}Starting FastAPI Backend...{RESET}")
    print(f"{BLUE}Command{RESET}: python -m uvicorn backends.rag_fastapi_server:app --reload --host localhost --port 8000")
    print(f"\n{BLUE}Server will be available at:{RESET} http://localhost:8000")
    print(f"{BLUE}API Documentation at:{RESET} http://localhost:8000/docs\n")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "backends.rag_fastapi_server:app", 
             "--reload", "--host", "localhost", "--port", "8000"],
            cwd=str(Path(__file__).parent)
        )
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Backend stopped by user{RESET}")
    except Exception as e:
        print(f"{RED}Error starting backend: {e}{RESET}")


def start_frontend():
    """Start Streamlit frontend"""
    print(f"\n{YELLOW}Starting Streamlit Frontend...{RESET}")
    frontend_path = Path(__file__).parent / "frontend" / "rag_streamlit_frontend.py"
    print(f"{BLUE}Command{RESET}: streamlit run {frontend_path}")
    print(f"\n{BLUE}Frontend will be available at:{RESET} http://localhost:8501")
    print(f"{BLUE}Make sure the backend is running on:{RESET} http://localhost:8000\n")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(frontend_path)],
            cwd=str(Path(__file__).parent)
        )
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Frontend stopped by user{RESET}")
    except Exception as e:
        print(f"{RED}Error starting frontend: {e}{RESET}")


def start_both():
    """Start both backend and frontend"""
    print(f"\n{YELLOW}Starting both Backend and Frontend...{RESET}\n")
    
    # Start backend in background
    print(f"{BLUE}[1/2]{RESET} Starting FastAPI Backend (http://localhost:8000)...")
    try:
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backends.rag_fastapi_server:app",
             "--reload", "--host", "localhost", "--port", "8000"],
            cwd=str(Path(__file__).parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"     {GREEN}✓ Backend process started (PID: {backend_process.pid}){RESET}")
    except Exception as e:
        print(f"     {RED}✗ Error starting backend: {e}{RESET}")
        return
    
    # Wait for backend to be ready
    print(f"\n{BLUE}Waiting for backend to be ready...{RESET}")
    time.sleep(3)
    
    # Start frontend
    print(f"\n{BLUE}[2/2]{RESET} Starting Streamlit Frontend (http://localhost:8501)...")
    frontend_path = Path(__file__).parent / "frontend" / "rag_streamlit_frontend.py"
    try:
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(frontend_path)],
            cwd=str(Path(__file__).parent)
        )
        print(f"     {GREEN}✓ Frontend started{RESET}\n")
    except Exception as e:
        print(f"     {RED}✗ Error starting frontend: {e}{RESET}")
        backend_process.terminate()
        return
    
    print(f"{GREEN}✓ Both services are running!{RESET}\n")
    print(f"{BOLD}Access the application:{RESET}")
    print(f"  - Frontend: http://localhost:8501")
    print(f"  - Backend: http://localhost:8000")
    print(f"  - API Docs: http://localhost:8000/docs")
    print(f"\n{YELLOW}Press Ctrl+C to stop both services{RESET}\n")
    
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Stopping both services...{RESET}")
        backend_process.terminate()
        backend_process.wait()
        print(f"{GREEN}✓ All services stopped{RESET}\n")


def check_api_health():
    """Check API health status"""
    print(f"\n{YELLOW}Checking API Health...{RESET}\n")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            print(f"{GREEN}✓ API is healthy!{RESET}\n")
            health_info = response.json()
            print(f"  Status: {health_info.get('status')}")
            print(f"  Service: {health_info.get('service')}")
            print(f"  Version: {health_info.get('version')}\n")
            
            # Try to get config
            try:
                config_response = requests.get("http://localhost:8000/config", timeout=5)
                if config_response.status_code == 200:
                    config = config_response.json()
                    print(f"{BOLD}Configuration:{RESET}")
                    print(f"  Pinecone: {config.get('pinecone_configured')}")
                    print(f"  OpenAI: {config.get('openai_configured')}")
                    print(f"  Index: {config.get('pinecone_index')}")
                    print()
            except:
                pass
        else:
            print(f"{RED}✗ API returned status {response.status_code}{RESET}\n")
    
    except requests.exceptions.ConnectionError:
        print(f"{RED}✗ Cannot connect to API at http://localhost:8000{RESET}")
        print(f"\n   Make sure the backend is running:")
        print(f"   {BLUE}python -m uvicorn backends.rag_fastapi_server:app --reload --host localhost --port 8000{RESET}\n")
    except ImportError:
        print(f"{RED}✗ requests library not installed{RESET}")
        print(f"   Install with: pip install requests\n")
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}\n")


def view_api_docs():
    """Open API documentation in browser"""
    print(f"\n{YELLOW}Opening API Documentation...{RESET}\n")
    
    try:
        import requests
        # Check if API is running
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            print(f"{GREEN}✓ Opening http://localhost:8000/docs in browser...{RESET}\n")
            webbrowser.open("http://localhost:8000/docs")
        else:
            print(f"{RED}✗ API not responding{RESET}\n")
    except:
        print(f"{RED}✗ Cannot connect to API at http://localhost:8000{RESET}")
        print(f"\n   Make sure the backend is running:")
        print(f"   {BLUE}python -m uvicorn backends.rag_fastapi_server:app --reload --host localhost --port 8000{RESET}\n")


def main():
    """Main menu loop"""
    print_header()
    
    while True:
        print_menu()
        choice = input(f"{BOLD}Enter your choice (1-6): {RESET}").strip()
        
        if choice == "1":
            start_backend()
        elif choice == "2":
            start_frontend()
        elif choice == "3":
            start_both()
        elif choice == "4":
            check_api_health()
        elif choice == "5":
            view_api_docs()
        elif choice == "6":
            print(f"\n{GREEN}Thank you for using MATLAB RAG Assistant!{RESET}\n")
            break
        else:
            print(f"{RED}Invalid choice. Please enter a number between 1 and 6.{RESET}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Application terminated by user{RESET}\n")
        sys.exit(0)
