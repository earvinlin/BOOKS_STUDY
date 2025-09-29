# 基於自行建立的SShd映像檔
FROM sshd:dockerfile
# 下面是一些建立者的基本資訊
MAINTAINER waitfish from dockerpool.com(earvinlin.03@gmail.com)
# 安裝nginx設置nginx以非daemon啟動
RUN \
apt-get install -y nginx && \
rm -rf /var/lib/apt/lists/* && \
echo "\ndaemon off;" >> /etc/nginx/nginx.conf && \ 
chown -R www-data:www-data /var/lib/nginx

RUN echo "Asia/Taipei" > /etc/timezone && \
dpkg-reconfigure -f noninteractive tzdata
# 注意這裡要更改系統的時區因為在Web應用程式經常會用到時區這個系統變數使用ubuntu預設的會讓你的應用程式發生不可預期的問題

# 增加run.Sh腳本並修改權限這會覆蓋掉之前放在這個位置的腳本檔
ADD run.sh /run.sh
RUN chmod 755 /*.sh

# 定義可以被掛載的目錄分別是虛擬主機的掛載目錄憑證目錄設定檔目錄和系統10g目錄
VOLUME ["/etc/nginx/sites-enabled", "/etc/nginx/certs", "/etc/nginx/conf.d", "/var/10g/nginx"]

# 設定工作目錄
WORKDIR /etc/nginx
# 定義執行指令
CMD ["/run.sh"]
# 定義開放連接埠
EXPOSE 80
EXPOSE 443

