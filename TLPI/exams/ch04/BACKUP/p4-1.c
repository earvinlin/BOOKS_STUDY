#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#define BUF_SIZE 1024

void usage(const char *progname) {
    fprintf(stderr, "使用方式：%s [-a] [-h] <filename>\n", progname);
    fprintf(stderr, "  -a    追加模式（append）\n");
    fprintf(stderr, "  -h    顯示使用說明\n");
}

int main(int argc, char *argv[]) {
    int opt;
    int append_mode = 0;
    int fd;
    char buffer[BUF_SIZE];
    ssize_t bytes;

    // 解析選項
    while ((opt = getopt(argc, argv, "ah")) != -1) {
        switch (opt) {
            case 'a':
                append_mode = 1;
                break;
            case 'h':
                usage(argv[0]);
                return 0;
            case '?':
                usage(argv[0]);
                return 1;
        }
    }

    // 檢查是否有指定檔案
    if (optind >= argc) {
        fprintf(stderr, "錯誤：請指定輸出檔案\n");
        usage(argv[0]);
        return 1;
    }

    const char *filename = argv[optind];

    // 設定開啟模式
    int flags = O_WRONLY | O_CREAT;
    flags |= append_mode ? O_APPEND : O_TRUNC;

    fd = open(filename, flags, 0644);
    if (fd == -1) {
        perror("開啟檔案失敗");
        return 1;
    }

    // 主迴圈：讀取 stdin → 同時寫入 stdout 與檔案
    while ((bytes = read(STDIN_FILENO, buffer, BUF_SIZE)) > 0) {
        write(STDOUT_FILENO, buffer, bytes);
        write(fd, buffer, bytes);
    }

    close(fd);
    return 0;
}
