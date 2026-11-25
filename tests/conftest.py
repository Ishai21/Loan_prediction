import os
import sys

# Make the project root importable so `import api` / `import loan_model` work
# regardless of where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
