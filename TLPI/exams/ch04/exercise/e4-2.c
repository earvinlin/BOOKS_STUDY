/*
    §4-2
    設計一個類似cp 指令的程式，當使用該程式複製一個包含空洞（連續的空位
    元組）的普通檔案時，要求目的檔案的空洞與原始檔案保持一致。
*/
#include <stdio.h>
#include <fcntl.h>      // open(), O_CREAT, O_WRONLY
#include <unistd.h>     // close()
#include <sys/stat.h>   // S_IRUSR, S_IWUSR 權限巨集
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif
#include <string.h>     // 使用 strrchr()

#define BUF_SIZE 30

int main(int argc, char *argv[]) {
    int inFd, outFd;
    char buffer[BUF_SIZE];
    ssize_t bytes_read;

    if (argc < 4) {
        printf("e4-2 file1 file2 copyType[0, 1]\n");
        exit(EXIT_FAILURE);
    }

    //=== 讀取檔案 ===//
    inFd = open(argv[1], O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (inFd == -1) {
        perror("open source!");
        exit(EXIT_FAILURE);
    }
    printf("成功開啟檔案，檔案描述符(FD)為：%d\n", inFd);

    //=== 寫入檔案 ===//
    outFd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (outFd == -1) {
        perror("open dest!");
        return 1; 
    }
    printf("成功建立檔案，檔案描述符(FD)為：%d\n", outFd);     

    if (atoi(argv[3]) == 0) {
        while ((bytes_read = read(inFd, buffer, BUF_SIZE)) > 0) {
            /*
            最關鍵原因：你的程式把「空洞」讀成 0 bytes，再用 write() 寫出去，所以目的檔案不再是 
            sparse file，而是被你寫成「實體的 0」填滿整個空洞區段，導致檔案大小變大。
            這段邏輯會：
            ．read() 讀到空洞 → 回傳一堆 0 bytes
            ．write() 把這些 0 bytes 寫到目的檔案
            ．原本的「洞」變成「真正寫入的 0」
            ．目的檔案不再是 sparse file
            ．檔案大小變成「實際寫入的 byte 數」，比原始檔案大
            */
            if (write(outFd, buffer, bytes_read) != bytes_read) {
                perror("write 寫入檔案失敗");
                close(inFd);
                return 1;
            }
        }
    } else {
        while ((bytes_read = read(inFd, buffer, BUF_SIZE)) > 0) {
            for (ssize_t i = 0; i < bytes_read; ) {
                /* 如果是 0 bytes，偵測連續空洞 */
                if (buffer[i] == 0) {
                    ssize_t holeStart = i;
                    while (i < bytes_read && buffer[i] == 0)
                        i++;
                    off_t holeSize = i - holeStart;
                    /* 用 lseek 跳過空洞，不寫 0 */
                    if (lseek(outFd, holeSize, SEEK_CUR) == -1) {
                        perror("lseek");
                        exit(EXIT_FAILURE);
                    }
                } else {
                    /* 非 0 bytes，正常寫出 */
                    ssize_t dataStart = i;
                    while (i < bytes_read && buffer[i] != 0)
                        i++;
                    ssize_t dataSize = i - dataStart;
                    if (write(outFd, &buffer[dataStart], dataSize) != dataSize) {
                        perror("write");
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
    }


    if (bytes_read == -1) {
        perror("read 發生錯誤");
         exit(EXIT_FAILURE);
    }
    printf("\n資料已成功寫入 %s！\n", argv[2]);

    close(inFd);  // 關閉檔案
    close(outFd);  // 關閉檔案

    return 0;
}
