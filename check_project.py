import subprocess
import sys


def main():
    print("Running project tests...")
    result = subprocess.run([sys.executable, "-m", "pytest", "-v"])

    if result.returncode == 0:
        print("\nProject validation passed.")
    else:
        print("\nProject validation failed.")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
