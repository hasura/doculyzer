#!/usr/bin/env python3

import subprocess
import sys


def update_version(version_type):
    """
    Executes the `bumpversion` command to update the version.

    :param version_type: Type of version bump ("major", "minor", or "patch").
    """
    valid_version_types = ["major", "minor", "patch"]

    # Validate the input
    if version_type not in valid_version_types:
        print(f"Error: Invalid version type '{version_type}'.")
        print("Valid options are: 'major', 'minor', or 'patch'.")
        sys.exit(1)

    try:
        # Execute the bumpversion command
        subprocess.run(["bumpversion", "--config-file", "pyproject.toml", version_type], check=True)
        print(f"Version successfully updated ({version_type}).")
    except FileNotFoundError:
        print("Error: `bumpversion` is not installed. Install it with:")
        print("  pip install bump2version")
    except subprocess.CalledProcessError:
        print("Error: Failed to update version. Check your `.bumpversion.cfg` configuration.")
        sys.exit(1)


if __name__ == "__main__":
    # Check if the user provided a version type
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version_type>")
        print("Example: python update_version.py minor")
        sys.exit(1)

    # Get the version type from the command-line arguments
    version_type = sys.argv[1].lower()

    # Call the function to update the version
    update_version(version_type)
