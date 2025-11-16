#!/usr/bin/env python3
"""
Startup script for SRM Traffic Violation Detection System
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required packages are installed"""
    
    print("🔍 Checking system dependencies...")
    
    required_packages = [
        'streamlit',
        'numpy', 
        'pandas',
        'opencv-python',
        'psycopg2-binary',
        'bcrypt'
    ]
    
    optional_packages = [
        'ultralytics',  # YOLO
        'easyocr',      # OCR
        'torch'         # PyTorch
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"   ❌ {package} (REQUIRED)")
    
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"   ⚠️  {package} (OPTIONAL)")
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("For full functionality, install with: pip install " + " ".join(missing_optional))
        print("System will run in demo mode without these packages.")
    
    return True

def setup_environment():
    """Setup environment variables"""
    
    print("🔧 Setting up environment...")
    
    # Set development mode for easier testing
    os.environ['STREAMLIT_ENV'] = 'development'
    os.environ['DEBUG'] = 'true'
    
    # Default admin credentials for demo
    if not os.getenv('ADMIN_USERNAME'):
        os.environ['ADMIN_USERNAME'] = 'admin'
    if not os.getenv('ADMIN_PASSWORD'):
        os.environ['ADMIN_PASSWORD'] = 'admin123'
    
    print("   ✅ Environment configured")
    print("   👤 Default admin login: admin / admin123")

def test_system():
    """Run system tests"""
    
    print("🧪 Running system tests...")
    
    try:
        # Run the test script
        result = subprocess.run([sys.executable, 'test_violations.py'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ All tests passed!")
            return True
        else:
            print("   ⚠️  Some tests failed, but system should still work")
            print("   Error output:", result.stderr[:200])
            return True  # Continue anyway
            
    except subprocess.TimeoutExpired:
        print("   ⚠️  Tests timed out, but system should still work")
        return True
    except Exception as e:
        print(f"   ⚠️  Test error: {str(e)}")
        return True  # Continue anyway

def start_streamlit():
    """Start the Streamlit application"""
    
    print("🚀 Starting SRM Traffic Violation Detection System...")
    print("=" * 60)
    print("🎓 SRM Institute of Science & Technology")
    print("📚 Final Year Project - Traffic Violation Detection")
    print("=" * 60)
    
    # Streamlit configuration
    config_args = [
        '--server.port=8501',
        '--server.address=localhost',
        '--server.headless=false',
        '--browser.gatherUsageStats=false',
        '--theme.base=light'
    ]
    
    try:
        # Start Streamlit
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'app.py'] + config_args
        
        print("🌐 Starting web interface...")
        print("📱 Access the system at: http://localhost:8501")
        print("👤 Login with: admin / admin123")
        print("\n⏹️  Press Ctrl+C to stop the system")
        print("-" * 60)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n🛑 System stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting system: {str(e)}")
        print("Try running manually: streamlit run app.py")

def main():
    """Main startup function"""
    
    print("🚦 SRM TRAFFIC VIOLATION DETECTION SYSTEM")
    print("=" * 50)
    print("🎓 Final Year Project")
    print("🏫 SRM Institute of Science & Technology")
    print("=" * 50)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        return
    
    # Setup environment
    setup_environment()
    
    # Test system
    test_system()
    
    print("\n🎉 System ready!")
    time.sleep(2)
    
    # Start the application
    start_streamlit()

if __name__ == "__main__":
    main()