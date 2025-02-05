# Utility script that installs mafft when it is not available on the system
# It requires git to be installed on the system
# Version will be bumped to latest version available on each revision

MAFFT_VERSION=v7.526;
echo "Installing MAFFT $MAFFT_VERSION";

PREFIX=$HOME/.local;

if ! test -d $PREFIX; then
    mkdir -p $PREFIX;
fi

mkdir -p mafft.temp;
cd mafft.temp;

# Clone the repository and checkout the required version
git clone https://gitlab.com/sysimm/mafft;
cd mafft/core;
git checkout $MAFFT_VERSION;

# Modify the Makefile to install MAFFT locally
sed -i "s|PREFIX = /usr/local|PREFIX = $PREFIX|" Makefile;
if [ $? -ne 0 ]; then
    echo "Failed to modify Makefile for MAFFT installation. Aborting...";
    exit 1;
fi

# Build and install
make clean;
make;
make install;

# Cleanup
cd ..;
rm -rf mafft.temp;

if ! test -f $HOME/.bashrc; then
    touch $HOME/.bashrc;
fi

if ! grep -q "PATH=$PREFIX/bin:\$PATH" $HOME/.bashrc; then
    echo "export PATH=$PREFIX/bin:\$PATH" >> $HOME/.bashrc;
fi
