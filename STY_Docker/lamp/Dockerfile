FROM tutum/lamp:latest
# FROM bitnami/lamp:latest
RUN rm -fr /app && git clone https://github.com/username/customapp.git /app
#這裡將https://github.com/username/customapp.git 位址替換為你自己的專案網址
EXPOSE 80 3306
CMD ["/run.sh"]


# § 11.5.3 部署自己的PHP應用
# 建立映像檔，命名為 dockerpool/my-lamp-app :
# docker build -t dockerpool/my-lamp-app .

# 新建立映像檔圈動容器，啟動時指定 -d 參，讓容器於背景執行 :
# docker run -d -p 8080:80 -p 3306:3306 dockerpool/my-lamp-app 

# 在本機上使用curl看一下自己的應用程式是不是已經正確啟動 :
# curl http://127.0.0.1:8080