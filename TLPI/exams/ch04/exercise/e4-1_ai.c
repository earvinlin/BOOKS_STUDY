#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#define BUF_SIZE 4096  /* 宣告緩衝區大小 (4KB) */

int main(int argc, char *argv[])
{
    int appendMode = 0; // 0: 覆蓋 (Truncate), 1: 追加 (Append)
    int opt;
    int openFlags;
    mode_t filePerms;
    int numRead;
    char buf[BUF_SIZE];

    /* --- 第一階段：使用 getopt() 解析命令列選項 --- */
    while ((opt = getopt(argc, argv, "a")) != -1) {
        switch (opt) {
        case 'a':
            appendMode = 1; // 使用者傳入了 -a 選項
            break;
        case '?':
        default:
            fprintf(stderr, "Usage: %s [-a] file...\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    /* 檢查是否至少傳入了一個檔名參數 */
    if (optind >= argc) {
        fprintf(stderr, "Usage: %s [-a] file...\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    /* --- 第二階段：決定 open() 的 Flag 設定 --- */
    /* 必備 Flags：唯寫 (O_WRONLY) + 檔名不存在則建立 (O_CREAT) */
    openFlags = O_WRONLY | O_CREAT;
    
    if (appendMode) {
        openFlags |= O_APPEND;  // -a 模式：附加在檔尾
    } else {
        openFlags |= O_TRUNC;   // 預設模式：清空覆蓋
    }

    /* 新建檔案時的權限設定 (rw-r--r--) */
    filePerms = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;

    /* --- 第三階段：開啟所有目標檔案 --- */
    int numFiles = argc - optind;
    int *fds = malloc(numFiles * sizeof(int));
    if (fds == NULL) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numFiles; i++) {
        // optind 是第一個非 Option 參數 (即檔名) 的 Index
        fds[i] = open(argv[optind + i], openFlags, filePerms);
        if (fds[i] == -1) {
            fprintf(stderr, "Error opening file %s: ", argv[optind + i]);
            perror("");
            // 這裡採取繼續處理其他檔案的策略
        }
    }

    /* --- 第四階段：核心 I/O 迴圈 (從 stdin 讀取，並輸出至 stdout 與各檔案) --- */
    while ((numRead = read(STDIN_FILENO, buf, BUF_SIZE)) > 0) {
        /* 1. 寫入標準輸出 (stdout) */
        if (write(STDOUT_FILENO, buf, numRead) != numRead) {
            perror("could not write whole buffer to stdout");
            exit(EXIT_FAILURE);
        }

        /* 2. 寫入所有成功開啟的檔案 */
        for (int i = 0; i < numFiles; i++) {
            if (fds[i] != -1) { // 確保檔案有成功開啟
                if (write(fds[i], buf, numRead) != numRead) {
                    fprintf(stderr, "could not write whole buffer to %s\n", argv[optind + i]);
                }
            }
        }
    }

    if (numRead == -1) {
        perror("read error from stdin");
        exit(EXIT_FAILURE);
    }

    /* --- 第五階段：資源回收 --- */
    for (int i = 0; i < numFiles; i++) {
        if (fds[i] != -1) {
            close(fds[i]);
        }
    }
    free(fds);

    exit(EXIT_SUCCESS);
}