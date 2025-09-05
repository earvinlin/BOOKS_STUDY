FROM ubuntu:22.04
LABEL maintainer="earvinlin.01@gmail.com"

# 更新套件來源為官方鏡像並安裝 openssh-server
RUN sed -i 's@http://.*.ubuntu.com@http://mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    apt-get update

RUN apt-get update
RUN apt-get install -y openssh-server
RUN mkdir -p /var/run/sshd /root/.ssh

# 複製 SSH 公鑰與啟動腳本
COPY authorized_keys /root/.ssh/authorized_keys
COPY run.sh /run.sh
RUN chmod 600 /root/.ssh/authorized_keys && \
    chmod 755 /run.sh

# 開放 SSH 連接埠
EXPOSE 22

# 設定容器啟動時執行的指令
CMD ["/run.sh"]

