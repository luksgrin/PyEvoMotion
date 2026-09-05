# For the data
FROM ghcr.io/luksgrin/pyevomotion-data:latest AS data

FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    sudo \
    python3-pip \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && apt-get clean

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

RUN useradd -ms /bin/bash pyevomotion && \
    usermod -aG sudo pyevomotion && \
    echo "pyevomotion ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to user and working dir
USER pyevomotion
WORKDIR /home/pyevomotion

# Add ~/.local/bin to PATH
ENV PATH="/home/pyevomotion/.local/bin:${PATH}"

# MAFFT is bundled as a pure-Rust extension inside the wheel — no external
# binary to install.

# Create venv and install PyEvoMotion inside it
RUN python3 -m venv /home/pyevomotion/venv && \
    /home/pyevomotion/venv/bin/pip install --upgrade pip && \
    /home/pyevomotion/venv/bin/pip download --no-deps --no-binary :all: \
        PyEvoMotion && \
    tar -xvzf pyevomotion-*.tar.gz && \
    cd pyevomotion-* && \
    /home/pyevomotion/venv/bin/pip install . pytest && \
    cd .. && \
    rm -rf pyevoMotion-* *.tar.gz

# Copy data from the data image and move to the correct location
COPY --from=data --chown=pyevomotion:pyevomotion /opt/data /home/pyevomotion/data
RUN mv /home/pyevomotion/data/* /home/pyevomotion/pyevomotion-*/tests/data/test3/.

# Copy fig data
COPY --from=data --chown=pyevomotion:pyevomotion /opt/figdata /home/pyevomotion/data
RUN mv /home/pyevomotion/data/* /home/pyevomotion/pyevomotion-*/share/. && \
    rmdir /home/pyevomotion/data


# Add venv to PATH
ENV PATH="/home/pyevomotion/venv/bin:$PATH"

# Welcome message
RUN echo '\n\
echo "\n\
👋 Welcome to the PyEvoMotion container!\n\nTo run PyEvoMotion, simply execute\n\n\
    PyEvoMotion\n\n\
followed by the appropriate arguments.\n\n\n\
To run the test suite execute the following commands:\n\n\
    cd pyevomotion-*\n\
    pytest\n\n\
To create the figure used in the original manuscript, execute the following commands:\n\n\
    cd pyevomotion-*\n\
    python share/manuscript_figure.py export\n\n\
Note that without the "export" argument, the figure will not be saved. \
Font errors may appear, but are harmless (stylistic only)."\n' \
>> /home/pyevomotion/.bashrc


CMD ["bash"]
