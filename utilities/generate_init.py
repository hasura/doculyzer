from tkinter import Tk
from tkinter.filedialog import askdirectory

import ast
import os


def extract_public_members(file_path):
    """
    Parse a Python file to extract public functions and classes.
    :param file_path: Path to the Python file.
    :return: A dictionary with keys 'functions' and 'classes' listing public names.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)

    public_functions = []
    public_classes = []

    # Traverse the AST (Abstract Syntax Tree) to find top-level definitions
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            public_functions.append(node.name)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            public_classes.append(node.name)

    return {"functions": public_functions, "classes": public_classes}


def generate_init_file(directory):
    """
    Automatically generates an __init__.py file with `__all__` and imports.
    Functions and classes take precedence over module names in case of conflicts.
    """
    init_file_path = os.path.join(directory, '__init__.py')

    # Prepare collections for __all__ and import statements
    all_exports = []
    import_statements = []

    # Process each Python file in the directory (skip __init__.py itself)
    for filename in os.listdir(directory):
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = filename[:-3]  # Strip the .py extension
            file_path = os.path.join(directory, filename)

            # Extract public functions and classes from the file
            members = extract_public_members(file_path)

            # Check for naming conflict: a module name matching a public function/class name
            conflict_detected = module_name in members["functions"] or module_name in members["classes"]

            if conflict_detected:
                print(
                    f"Conflict detected: Skipping module import for '{module_name}'. Public method/class will take priority.")

            # Add each function and class to __all__ and import statements
            for func in members["functions"]:
                # Prioritize public function over module
                all_exports.append(func)
                import_statements.append(f"from .{module_name} import {func}")

            for cls in members["classes"]:
                # Prioritize public class over module
                all_exports.append(cls)
                import_statements.append(f"from .{module_name} import {cls}")

            # Import the module itself only if there's no naming conflict
            if not conflict_detected:
                all_exports.append(module_name)
                import_statements.append(f"from . import {module_name}")

    # Eliminate duplicates in `__all__` (if any)
    all_exports = sorted(set(all_exports))

    # Write the __init__.py content
    with open(init_file_path, 'w', encoding="utf-8") as init_file:
        init_file.write("\"\"\"Automatically generated __init__.py\"\"\"\n")
        init_file.write(f"__all__ = {all_exports}\n\n")
        init_file.write("\n".join(import_statements))
        init_file.write("\n")

    print(f"`__init__.py` generated successfully at: {init_file_path}")


if __name__ == "__main__":
    # Use tkinter to open a folder picker
    Tk().withdraw()  # Hide the root window
    directory = askdirectory(title="Select a directory for __init__.py generation")

    if not directory:
        print("No directory selected. Exiting.")
    elif not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
    else:
        generate_init_file(directory)
