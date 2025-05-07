import os


def get_vendor_path():
    """
    Dynamically determines the correct path to the vendor directory based on runtime context.

    Returns:
        str: Absolute path to the vendor directory
    """
    # Determine the current module directory
    package_dir = os.path.dirname(os.path.abspath(__file__))

    # Debug mode: Check if two levels up contains `vendor/libmagic`
    debug_vendor_path = os.path.abspath(os.path.join(package_dir, '..', '..', 'vendor', 'libmagic'))

    # Installed mode: Check if `vendor/libmagic` exists in the current directory
    installed_vendor_path = os.path.join(package_dir, 'vendor', 'libmagic')

    # Prioritize debug path if it exists
    if os.path.exists(debug_vendor_path):
        return debug_vendor_path

    # Fallback to the installed path
    if os.path.exists(installed_vendor_path):
        return installed_vendor_path

    # If neither path exists, raise an error
    raise FileNotFoundError("The 'vendor/libmagic' directory could not be located.")
