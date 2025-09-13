# 使用官方 Debian 基礎映像
FROM debian:bullseye

# 安裝 Apache HTTP Server
RUN apt-get update && \
    apt-get install -y apache2 && \
    apt-get clean

# Marked 2025.09.12
RUN mkdir -p /var/lock/apache2 && mkdir -p /app && rm -fr /var/www/html && ln -s /app /var/www/html
COPY sample/ /app
WORKDIR /app

# 開放 80 port
EXPOSE 80

# 啟動 Apache（前景模式）
CMD ["apachectl", "-D", "FOREGROUND"]