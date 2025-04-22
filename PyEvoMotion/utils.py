import os
import sys
import shutil
import subprocess


def get_mafft_script_path() -> str:
    # Find the installed location of this package
    package_root = os.path.dirname(sys.modules["PyEvoMotion"].__file__)

    share_path = os.path.abspath(os.path.join(package_root, "..", "share", "mafft_install.sh"))

    if not os.path.exists(share_path):
        raise FileNotFoundError(f"mafft_install.sh not found at {share_path}")

    return share_path

def ensure_local_bin_in_path() -> bool:
    """Checks if ~/.local/bin is in the PATH environment variable."""
    local_bin = os.path.expanduser("~/.local/bin")
    if local_bin not in os.environ.get("PATH", ""):
        print(f"\n⚠️  {local_bin} is not in your PATH.")
        print("You may not be able to run 'mafft' from the terminal.")
        print("To fix this, add the following line to your shell config (e.g., ~/.bashrc, ~/.zshrc):\n")
        print(f'    export PATH="$PATH:{local_bin}"\n')
        print("Then restart your shell or run `source ~/.bashrc`.\n")
        return False
    return True

def get_mafft_path() -> str:
    """Returns the path to the mafft binary if found, else None."""
    return shutil.which("mafft") or os.path.expandvars("$HOME/.local/bin/mafft")

def is_mafft_installed() -> bool:
    """Returns True if mafft is available."""
    return shutil.which("mafft") is not None or os.path.exists(os.path.expandvars("$HOME/.local/bin/mafft"))

def install_mafft():

    response = input(
        "mafft is not installed. Would you like to install it locally? (y/n): "
    ).strip().lower()
    if response not in ["y", "yes"]:
        print("mafft installation aborted.")
        exit(0)

    """Installs mafft locally using the bundled script."""
    print("Installing mafft locally...")

    try:
        subprocess.run(["bash", get_mafft_script_path()], check=True)
        print("mafft installation complete.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install mafft: {e}")
        sys.exit(1)

def verify_mafft():
    """Runs `mafft --version` to confirm installation."""

    if not ensure_local_bin_in_path():
        sys.exit(0)

    mafft_path = get_mafft_path()
    if not mafft_path:
        print("mafft not found.")
        return False

    try:
        result = subprocess.run(
            [mafft_path, "--version"],
            capture_output=True,
            text=True
        )
        print(f"mafft version: {result.stderr.strip() or result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"Error running mafft: {e}")
        return False

def check_and_install_mafft():
    if not is_mafft_installed():
        print("mafft not found.")
        install_mafft()

        if not verify_mafft():
            print("mafft verification failed after installation.")
            sys.exit(1)
