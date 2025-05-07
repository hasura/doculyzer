import os


def get_vendor_path():
    """
    Dynamically determines the correct path to the vendor directory based on runtime context.

    Returns:
        str: Absolute path to the vendor directory
    """
    # Determine the current module directory
    package_dir = os.path.dirname(os.path.abspath(__file__))
    vendor_path = os.path.join(package_dir, 'vendor', 'libmagic')
    if os.path.exists(vendor_path):
        return vendor_path
    raise FileNotFoundError(f"The 'vendor/libmagic' directory could not be located.: {vendor_path}")
