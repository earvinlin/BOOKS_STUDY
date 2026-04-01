FROM ubuntu:22.04
LABEL maintainer="earvin@localhost.com"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y apache2 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/share/zoneinfo/Asia/Taipei /etc/localtime && \
    echo "Asia/Taipei" > /etc/timezone

ADD run.sh /run.sh
RUN chmod 755 /*.sh

RUN mkdir -p /var/lock/apache2 && mkdir -p /app && \
    rm -fr /var/www/html && ln -s /app /var/www/html
COPY sample/ /app

RUN mkdir -p /var/run/apache2 

ENV APACHE_RUN_USER=www-data
ENV APACHE_RUN_GROUP=www-data
ENV APACHE_LOG_DIR=/var/log/apache2
#ENV APACHE_PID_FILE=/var/run/apache2.pid
ENV APACHE_PID_FILE=/var/run/apache2/apache2.pid
ENV APACHE_RUN_DIR=/var/run/apache2
ENV APACHE_LOCK_DIR=/var/lock/apache2
ENV APACHE_SERVERADMIN=admin@localhost
ENV APACHE_SERVERNAME=localhost
ENV APACHE_SERVERALIAS=docker.localhost
ENV APACHE_DOCUMENTROOT=/var/www

EXPOSE 80
WORKDIR /app
CMD ["/run.sh"]

