FROM tensorflow/tensorflow:2.4.0-gpu-jupyter

ENV DEBIAN_FRONTEND noninteractive

# install essentials
RUN apt-get update -y && apt-get install -y \
	software-properties-common \
	build-essential \
	libblas-dev \
	libhdf5-serial-dev \
	git

# install some niceties
RUN apt-get update -y && apt-get install -y zsh tmux htop vim
RUN pip3 install -U pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# make terminals look pretty (setting a reasonable colour setting)
RUN touch /usr/share/locale/locale.alias
RUN apt-get -y install locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV TERM xterm-256color

# change workdir and add some files
WORKDIR /src
ADD ./requirements.txt ./
ADD ./git-requirements.txt ./

# install our beloved requirements
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN pip install -r git-requirements.txt

# extra stuff that I needed along the way
RUN apt-get update -y && apt-get install -y graphviz --fix-missing
RUN apt-get update -y && apt-get install -y jupyter
RUN apt-get update -y && apt-get install -y python3-venv
RUN apt-get autoremove -y && apt-get autoclean -y