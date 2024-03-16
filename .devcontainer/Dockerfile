FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# update the ubuntu os image and install mandatory libs.
# CAUTION: Don't upgrade packages, especially the CUDA and CuDNN.
RUN apt update &&  \
    apt install -y \
        sudo wget curl apt-utils locales bash-completion fonts-noto-cjk && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# install python library
RUN /usr/local/bin/pip3 install --upgrade pip && \
    /usr/local/bin/pip3 install \
    autopep8 black yapf bandit flake8 mypy pycodestyle pydocstyle pylint \
    jupyterlab-language-pack-ja-JP keras-tqdm pillow pandas bokeh matplotlib && \
    /usr/local/bin/pip3 cache purge

# create vscode user
ARG USERNAME=vscode
ARG GROUPNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd -g 1000 ${GROUPNAME} && \
    useradd -m -s /bin/bash -u ${USER_UID} -g ${USER_GID} ${USERNAME} && \
    echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

# install poerty for vscode user
USER vscode
ARG WORKDIR
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN echo "export PATH=/home/vscode/.local/bin:$PATH" >> /home/vscode/.bashrc
RUN /home/vscode/.local/bin/poetry completions bash | sudo tee /etc/bash_completion.d/poerty.bash-completion > /dev/null
