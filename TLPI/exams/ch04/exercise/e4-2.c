/*
    §4-2
    設計一個類似cp 指令的程式，當使用該程式複製一個包含空洞（連續的空位
    元組）的普通檔案時，要求目的檔案的空洞與原始檔案保持一致。
*/
#include <stdio.h>
#include <fcntl.h>   // open(), O_CREAT, O_WRONLY
#include <unistd.h>  // close()
#include <sys/stat.h> // S_IRUSR, S_IWUSR 權限巨集
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
#include <string.h> // 使用 strrchr()

#define BUF_SIZE 30

int main(int argc, char *argv[]) {
    int fd1, fd2;
    char buffer[BUF_SIZE];
    ssize_t bytes_read;

    if (argc < 3) {
        printf("e4-2 file1 file2 \n");
        exit(1);
    }
    fd1 = open(argv[1], O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd1 == -1) {
        perror("open fail! (for read)");
        return 1;
    }
    printf("成功開啟檔案，檔案描述符(FD)為：%d\n", fd1);
            
    while ((bytes_read = read(fd1, buffer, BUF_SIZE)) > 0) {
        // 將讀取到的資料寫入 Terminal (標準輸出 STDOUT_FILENO = 1)
        if (write(STDOUT_FILENO, buffer, bytes_read) != bytes_read) {
            perror("write 寫入螢幕失敗");
            close(fd1);
            return 1;
        }
    }

    if (bytes_read == -1) {
        perror("read 發生錯誤");
    } else {
        printf("\n=== 檔案讀取完畢 (EOF) ===\n");
    }

    close(fd1);  // 關閉檔案

    return 0;
}
