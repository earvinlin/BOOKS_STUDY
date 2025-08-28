#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

#define BUF_SIZE 4096

int is_all_zero(char *buf, ssize_t size) {
    for (ssize_t i = 0; i < size; i++) {
        if (buf[i] != 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "使用方式：%s <來源檔> <目的檔>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int src_fd = open(argv[1], O_RDONLY);
    if (src_fd < 0) {
        perror("開啟來源檔失敗");
        exit(EXIT_FAILURE);
    }

    int dst_fd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (dst_fd < 0) {
        perror("開啟目的檔失敗");
        close(src_fd);
        exit(EXIT_FAILURE);
    }

    char buffer[BUF_SIZE];
    off_t offset = 0;

    while (1) {
        ssize_t bytes = read(src_fd, buffer, BUF_SIZE);
        if (bytes < 0) {
            perror("讀取錯誤");
            break;
        }
        if (bytes == 0) break;  // EOF

        if (is_all_zero(buffer, bytes)) {
            // 建立空洞：移動目的檔案指標
            lseek(dst_fd, bytes, SEEK_CUR);
        } else {
            // 寫入實際資料
            ssize_t written = write(dst_fd, buffer, bytes);
            if (written != bytes) {
                perror("寫入錯誤");
                break;
            }
        }
        offset += bytes;
    }

    close(src_fd);
    close(dst_fd);
    return 0;
}
