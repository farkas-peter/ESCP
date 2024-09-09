FROM sanluosizhou/selfdl:ml

RUN echo "fs.inotify.max_user_watches = 524288" >> /etc/sysctl.conf
RUN sysctl -p --system

RUN apt-key del 7fa2af80
COPY ./cuda-keyring_1.0-1_all.deb .
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-key del 7fa2af80
#RUN rpm --erase gpg-pubkey-7fa2af80*
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

ENV LANG=en_US.UTF-8
RUN apt-get update -qq && apt-get install -qy gnome-terminal libcanberra-gtk-module libcanberra-gtk3-module locales
RUN echo 'LANG=en_US.UTF-8' > '/etc/default/locale' && \
    locale-gen --lang en_US.UTF-8 && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=$LANG

# Install make and compilers and extra stuff
RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qy

RUN apt update -qq && apt install -qy openssh-server sudo vim git curl wget tmux gcc cmake gdb build-essential

ENTRYPOINT /entry.sh


