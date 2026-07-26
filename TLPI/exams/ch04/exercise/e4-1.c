/*
    §4-1
    tee 指令會從標準輸入中讀取資料，直至檔案結尾，隨後將資料寫入標準輸出和命
    令列參數所指定的檔案。請使用 I/O 系統呼叫實作 tee 指令。預設情況下，若已
    存在與命令列參數指定檔案同名的檔案，則tee 指令會將其覆蓋。如檔案已存在，
    請實作 -a 命令列選項（tee -a file）在檔案結尾處追加資料。（請參考附錄B 
    中對getopt（）函式的描述來解析命令列選項。）
    
    # 1. 編譯程式碼
    gcc -Wall e4-1.c -o e4-1_arm

    # 2. 測試預設（覆蓋）模式
    echo "Hello, World!" | ./e4-1_arm output.txt
    echo "Hello, World!" | ./e4-1_intel output_a.txt output_b.txt
    # 螢幕會印出 Hello, World!，且 output.txt 內容為 "Hello, World!"

    # 3. 測試 -a（追加）模式
    echo "Second Line" | ./e4-1_arm -a output.txt
    echo "Second Line" | ./e4-1_intel -a output_a.txt output_b.txt
    # 螢幕會印出 Second Line，且 output.txt 內容變為 2 行！
    cat output.txt
*/
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#if defined(USE_MYLIB_INTEL)
#include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"
#else
#include "../../../tlpi-book/mylib/tlpi_hdr.h"
#endif

#include <string.h>

#define BUF_SIZE 4096

int main(int argc, char *argv[])
{
    int appendMode = 0; // 0: 覆蓋 (Truncate), 1: 追加 (Append)
    int opt;
    int openFlags;
    mode_t filePerms;
    ssize_t numRead;
    char buf[BUF_SIZE];

    /* --- 使用 getopt() 解析命令列選項 --- */
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
    // getopt 處理完所有的 -a 等選項後，optind 全域變數會指向 argv 中**第一個
    // 「非選項」參數（也就是目標檔名 file.txt）**的位置。
    if (optind >= argc) {
        fprintf(stderr, "Usage: %s [-a] file...\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    if (appendMode) {
        openFlags = O_WRONLY | O_CREAT | O_APPEND; // -a 模式：附加在檔尾
    } else {
        openFlags = O_WRONLY | O_CREAT | O_TRUNC; // 預設模式：清空覆蓋
    }
    /* 新建檔案時的權限設定 (rw-r--r--) */
    filePerms = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;

    // 開啟所有目標檔案
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

    // 從Terminal標準輸入讀取資料，並寫入標準輸出和所有成功開啟的檔案
    while ((numRead = read(STDIN_FILENO, buf, BUF_SIZE)) > 0) {
        // 寫入標準輸出 (stdout)
        if (write(STDOUT_FILENO, buf, numRead) != numRead) {
            perror("could not write whole buffer to stdout");
            exit(EXIT_FAILURE);
        }

        // 寫入所有成功開啟的檔案 
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

    for (int i = 0; i < numFiles; i++) {
        if (fds[i] != -1) {
            close(fds[i]);
        }
    }
    free(fds);  // 因為是用malloc()配置的記憶體，所以要用free()釋放

/*
    for (int i = optind; i < argc; i++) {
        outFd[i] = open(argv[i], openFlags, filePerms);
        if (outFd[i] == -1)
            errExit("open dest");
        // write out 
        while ((numRead = read(inFd, buf, BUF_SIZE)) > 0) {
            if (write(outFd[i], buf, numRead) != numRead)
                errExit("write");
        }
    }
*/

    return 0;
}
