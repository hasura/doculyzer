import os
import shutil


def build_and_upload():
    """
    Build the project and upload to PyPI.
    """
    try:
        # Step 1: Clear the `dist/` directory
        dist_dir = "dist"
        if os.path.exists(dist_dir):
            shutil.rmtree(dist_dir)  # Remove the directory and its contents
        os.makedirs(dist_dir)  # Recreate an empty `dist` directory

        # Step 2: Build the distribution
        subprocess.run(["python", "-m", "build"], check=True)
        print("Build completed successfully.")
    except subprocess.CalledProcessError:
        print("Error: Failed to build the distribution.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    try:
        # Step 3: Upload to PyPI using Twine
        subprocess.run(["python", "-m", "twine", "upload", "dist/*"], check=True)
        print("Package uploaded successfully to PyPI.")
    except subprocess.CalledProcessError:
        print("Error: Failed to upload package to PyPI. Check your Twine configuration.")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: `twine` is not installed. Install it with:")
        print("  pip install twine")
        sys.exit(1)


if __name__ == "__main__":
    # Check if the user provided a version type
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version_type>")
        print("Example: python update_version.py minor")
        sys.exit(1)

    # Get the version type from the command-line arguments
    version_type = sys.argv[1].lower()

    # Step 1: Update the version
    update_version(version_type)

    # Step 2: Build and Upload
    build_and_upload()
