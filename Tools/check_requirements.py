import importlib

# Adjusted to match actual import names
required = [
    'pandas', 'numpy', 'sklearn', 'matplotlib',
    'seaborn', 'flask', 'joblib'
]

print("🔍 Checking required packages:\n")

for pkg in required:
    try:
        importlib.import_module(pkg)
        print(f"✅ {pkg} is installed.")
    except ImportError:
        print(f"❌ {pkg} is NOT installed.")
        print(f"Please install it using: pip install {pkg}\n")