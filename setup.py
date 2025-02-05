import os, sys, subprocess
from setuptools import setup
from setuptools.command.install import install

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

class MafftCheck(install):

    @classmethod
    def __check_mafft(cls):

        # Check if mafft is installed
        try:
            cls.__get_mafft_version("mafft") # If installed, this will print the version

        except FileNotFoundError:
            print("mafft not found. Installing mafft locally...")

            try:
                subprocess.run(["bash", "share/mafft_install.sh"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error installing mafft: {e.output}.")
                sys.exit(1)

            # Check if mafft was installed successfully
            try:
                cls.__get_mafft_version(os.path.expandvars(
                    "$HOME/.local/bin/mafft"
                )) # If installed, this will print the version
                print("mafft installation successful.")
            except subprocess.CalledProcessError:
                print("Error checking mafft installation.")
                sys.exit(1)
        except subprocess.CalledProcessError:
            print("Error installing mafft.")
            sys.exit(1)

    @staticmethod
    def __get_mafft_version(executable: str):
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True
        )
        print(f"mafft found: {result.stderr}")

    def run(self):
        self.__check_mafft()
        install.run(self)

setup(
    name = "PyEvoMotion",
    version = "0.1.0",
    author = "Lucas Goiriz",
    author_email = "lucas.goiriz@csic.es",
    description = read('README.md'),
    # license = "BSD",
    # keywords = "example documentation tutorial",
    # url = "http://packages.python.org/an_example_pypi_project",
    packages=['PyEvoMotion', 'PyEvoMotion/core'],
    data_files=[('share', ['share/mafft_install.sh'])],
    long_description=read('README.md'),
    python_requires=">=3.10,<3.13",
    install_requires=[
        "pytest>=8.2.2,<9.0.0",
        "bio>=1.7.1,<2.0.0",
        "pandas>=2.2.2,<3.0.0",
        "scikit-learn>=1.5.1,<2.0.0",
        "matplotlib>=3.9.1,<4.0.0"
    ],
    entry_points={
        'console_scripts': [
            'PyEvoMotion=PyEvoMotion.cli:_main',
        ],
    },
    cmdclass={
        'install': MafftCheck,
    },
    # classifiers=[
    #     "Development Status :: 3 - Alpha",
    #     "Topic :: Utilities",
    #     "License :: OSI Approved :: BSD License",
    # ],
)

print("Setup finished.")
print("If mafft was not installed in your system, it was installed locally. Restart your shell to make it available for PyEvoMotion.")