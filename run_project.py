"""
Seed Analysis System - Project Runner
This script starts the project from any location, automatically setting correct paths and environment variables
"""

import os
import sys
import subprocess

# Get absolute path to project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add project root directory to Python path
sys.path.insert(0, ROOT_DIR)

# Set environment variables
os.environ["PYTHONPATH"] = ROOT_DIR
os.environ["PYTHONHASHSEED"] = "123"

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []

    try:
        import torch
    except ImportError:
        missing_deps.append("torch")

    try:
        import fastapi
    except ImportError:
        missing_deps.append("fastapi")

    try:
        import uvicorn
    except ImportError:
        missing_deps.append("uvicorn")

    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    try:
        import PIL
    except ImportError:
        missing_deps.append("pillow")

    if missing_deps:
        print("=" * 50)
        print("ERROR: Missing dependencies")
        print("=" * 50)
        print("The following packages are required but not installed:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_deps)}")
        print("or")
        print("pip install -r requirements.txt")
        print("=" * 50)
        return False

    return True

def setup_pytorch():
    """Setup PyTorch environment"""
    import torch

    # Set random seed
    torch.manual_seed(123)

    # Configure CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set thread count
    torch.set_num_threads(1)

    return torch

def main():
    """Main function to start the project"""
    # Check dependencies
    if not check_dependencies():
        return 1

    # Setup PyTorch
    try:
        torch = setup_pytorch()
    except Exception as e:
        print(f"Error setting up PyTorch: {str(e)}")
        return 1

    # Set backend directory and main script path
    backend_dir = os.path.join(ROOT_DIR, "backend")
    main_script = os.path.join(backend_dir, "main.py")

    # Change to backend directory
    os.chdir(backend_dir)

    # Print environment information
    print("=" * 50)
    print("Seed Analysis System Starting")
    print("=" * 50)
    print(f"Python interpreter: {sys.executable}")
    print(f"Project root: {ROOT_DIR}")
    print(f"Working directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 50)

    # Run backend
    print(f"Starting backend service: {main_script}")
    try:
        subprocess.run([sys.executable, main_script])
        return 0
    except KeyboardInterrupt:
        print("\nService stopped")
        return 0
    except Exception as e:
        print(f"Error running backend: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
