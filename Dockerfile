# Use the specified Ubuntu image
FROM ubuntu:noble-20240114

# Install prerequisites including python3-venv
RUN apt-get update
RUN apt-get install -y \
    build-essential \
    gcc \
    python3 \
    python3-pip \
    python3-venv \
    libblas-dev \
    liblapack-dev \
    python3-dev \
    git \
    tree \
    gfortran

# Create a virtual environment and activate it
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python packages in the virtual environment
RUN pip install numpy tensorflow

# Copy your DCI project into the Docker image
WORKDIR /opt/dci
COPY . /opt/dci

# Compile and install the DCI library
RUN python setup.py install

# verify installation
# RUN python3 examples/example.py

CMD ["bash"]