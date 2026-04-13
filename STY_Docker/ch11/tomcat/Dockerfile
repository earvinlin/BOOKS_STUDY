FROM sshd:dockerfile

#基於自行建立的5shd映像檔
MAINTAINER waitfish from dockerpool.com(dwj_zz@163.com)

#下面是一些建立者的基本資訊

# 設定環境變數 改為非互動式操作
ENV DEBIAN_FRONTEND noninteractive

RUN echo "Asia/Taipei" › /etc/timezone && dkg-reconfigure -f noninteractive tzdata

RUN apt-get install -yq --no-install-recommends wet pwen ca-certificates && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

# 設定tomcat的環境變數 若您有其他的環境變數需要設定 也可以在這裡加入
ENV CATALINA_HOME /tomcat
ENV JAVA_HOME /jdk

# 複製tomcat和jdk檔到映像檔中
ADD apache-tomcat-7.0.59 /tomcat
ADD jdk /jdk

ADD create_tomcat_admin_user.sh /create_tomcat _admin_user.sh
ADD run.sh /run.sh
RUN chmod +x /*.sh
RUN chmod +x /tomcat/bin/*.sh

EXPOSE 8080

CMD ["/run.sh"]

