#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <sys/socket.h>
#include <libgen.h> // 使用 basename() 必須引入此標頭檔

int main(int argc, char *argv[]) {
    int fd;
    struct ifreq ifr;

    char * netCardName;

    if (argc < 2) {
        char *prog_name = basename(argv[0]);    // 使用 basename 取得不帶路徑的純檔名
        printf("parameter error, program is %s\n", argv[0]);

        exit(1);
    }

    netCardName = argv[1];

    // 1. 開啟一個 socket 作為硬體控制介面
    fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        perror("socket 建立失敗");
        return 1;
    }

    // 2. 設定要查詢的網卡名稱 (例如 eth0 或 wlan0)
//    strncpy(ifr.ifr_name, "eth0", IFNAMSIZ - 1);
    strncpy(ifr.ifr_name, netCardName, IFNAMSIZ - 1);

    // 3. 呼叫 ioctl，傳入 SIOCGIFHWADDR (Get Hardware Address 命令)
    if (ioctl(fd, SIOCGIFHWADDR, &ifr) < 0) {
        perror("ioctl 查詢失敗");
        close(fd);
        return 1;
    }

    close(fd);

    // 4. 解析結構體中的 MAC 位址
    unsigned char *mac = (unsigned char *)ifr.ifr_hwaddr.sa_data;
    printf("%s 的 MAC 位址為: %02x:%02x:%02x:%02x:%02x:%02x\n",
           argv[1], mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    return 0;
}
