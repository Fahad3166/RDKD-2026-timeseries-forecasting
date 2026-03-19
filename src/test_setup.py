# =============================================================================
# src/test_setup.py
# script to verify your project setup
# =============================================================================

import sys
from pathlib import Path

print("="*60)
print("TESTING PROJECT SETUP")
print("="*60)

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
print(f"\n📁 Script directory: {script_dir}")

# Get the project root (parent of src directory)
project_root = script_dir.parent
print(f"📁 Project root: {project_root}")

# Add project root to Python path
sys.path.insert(0, str(project_root))
print(f"✅ Added {project_root} to Python path")

# Check Python version
print(f"\n🐍 Python version: {sys.version}")

# Check current working directory
cwd = Path.cwd()
print(f"\n📁 Current working directory: {cwd}")

# Now try to import config
print("\n" + "-"*40)
print("Testing imports...")
print("-"*40)

try:
    from src.config import PROJECT_ROOT as CONFIG_ROOT, RAW_DATA_DIR
    print(f"✅ Successfully imported config")
    print(f"   Config's PROJECT_ROOT: {CONFIG_ROOT}")
    print(f"   Raw data dir: {RAW_DATA_DIR}")
    
    # Check if raw data directory exists
    if RAW_DATA_DIR.exists():
        print(f"   ✅ Raw data directory exists")
        
        # Check for CSV files
        csv_files = list(RAW_DATA_DIR.glob('*.csv'))
        if csv_files:
            print(f"   📄 CSV files found:")
            for f in csv_files:
                size = f.stat().st_size
                print(f"      - {f.name} ({size} bytes)")
        else:
            print(f"   ❌ No CSV files found in {RAW_DATA_DIR}")
            print(f"      Please place sample_23.csv and sample_24.csv here")
    else:
        print(f"   ❌ Raw data directory does not exist: {RAW_DATA_DIR}")
        
except ImportError as e:
    print(f"❌ Failed to import config: {e}")
    print(f"   Python path: {sys.path}")

print("\n" + "-"*40)

try:
    from src.data_loader import load_data, quick_check
    print(f"✅ Successfully imported data_loader")
    
    # Try to load data (with verbose=False to keep output clean)
    print(f"\n📊 Attempting to load data...")
    df_23, df_24 = load_data(verbose=False)
    
    # Quick check
    info = quick_check(df_23, df_24)
    print(f"\n📊 Data quick check:")
    for key, value in info.items():
        print(f"   - {key}: {value}")
    
    # Show first few rows of 2023 data
    print(f"\n📋 First 3 rows of 2023 data:")
    print(df_23.head(3))
    
except ImportError as e:
    print(f"❌ Failed to import data_loader: {e}")
except FileNotFoundError as e:
    print(f"❌ File not found error: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("SETUP TEST COMPLETE")
print("="*60)