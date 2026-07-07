"""Script to detect circular imports in the project.

This script attempts to import each module in the project and reports any
circular import errors that occur during the process.
"""
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Set, Tuple

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Track modules that have been checked
checked_modules: Set[str] = set()
# Track modules that are currently being imported (to detect cycles)
importing_modules: Set[str] = set()
# Track detected circular imports
circular_imports: Dict[str, List[str]] = {}

def is_python_file(path: Path) -> bool:
    """Check if a path is a Python file."""
    return path.is_file() and path.suffix == '.py'

def is_python_package(path: Path) -> bool:
    """Check if a path is a Python package."""
    return path.is_dir() and (path / '__init__.py').exists()

def module_path_to_name(path: Path, root: Path) -> str:
    """Convert a file path to a module name."""
    rel_path = path.relative_to(root)
    if path.name == '__init__.py':
        rel_path = rel_path.parent
    else:
        rel_path = rel_path.with_suffix('')
    return str(rel_path).replace(os.sep, '.')

def find_modules(root: Path) -> List[str]:
    """Find all Python modules in the project."""
    modules = []
    for path in root.glob('**/*.py'):
        if '__pycache__' in path.parts:
            continue
        module_name = module_path_to_name(path, root)
        modules.append(module_name)
    return modules

def try_import_module(module_name: str) -> Tuple[bool, str]:
    """Try to import a module and catch circular import errors."""
    if module_name in checked_modules:
        return True, ""
    
    if module_name in importing_modules:
        return False, f"Circular import detected: {module_name} is already being imported"
    
    importing_modules.add(module_name)
    try:
        importlib.import_module(module_name)
        checked_modules.add(module_name)
        importing_modules.remove(module_name)
        return True, ""
    except ImportError as e:
        importing_modules.remove(module_name)
        if "circular import" in str(e).lower():
            return False, str(e)
        # Other import errors might be due to missing dependencies, not circular imports
        checked_modules.add(module_name)
        return True, ""
    except Exception as e:
        importing_modules.remove(module_name)
        return False, f"Error importing {module_name}: {str(e)}"

def detect_circular_imports() -> Dict[str, List[str]]:
    """Detect circular imports in the project."""
    modules = find_modules(project_root)
    print(f"Found {len(modules)} modules to check")
    
    for module_name in modules:
        success, error = try_import_module(module_name)
        if not success:
            if "circular import" in error.lower():
                circular_imports[module_name] = [error]
                print(f"Circular import in {module_name}: {error}")
    
    return circular_imports

if __name__ == "__main__":
    print("Detecting circular imports...")
    circular_imports = detect_circular_imports()
    
    if circular_imports:
        print("\nCircular imports detected:")
        for module, errors in circular_imports.items():
            print(f"\n{module}:")
            for error in errors:
                print(f"  - {error}")
        sys.exit(1)
    else:
        print("No circular imports detected!")
        sys.exit(0)