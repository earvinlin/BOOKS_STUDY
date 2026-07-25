/* §4-2 設計一個類似 cp 的程式，能保留 sparse file 的空洞 */

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

/* ---------------------------------------------------------
 * 一般複製：read → write
 * --------------------------------------------------------- */
static void
copy_normal(int inFd, int outFd)
{
    char buf[BUF_SIZE];
    ssize_t numRead;

    while ((numRead = read(inFd, buf, BUF_SIZE)) > 0) {
        if (write(outFd, buf, numRead) != numRead)
            errExit("write");
    }

    if (numRead == -1)
        errExit("read");
}

/* ---------------------------------------------------------
 * sparse-aware 複製：偵測 0 bytes → lseek 跳過空洞
 * --------------------------------------------------------- */
static void
copy_sparse(int inFd, int outFd)
{
    char buf[BUF_SIZE];
    ssize_t numRead;

    while ((numRead = read(inFd, buf, BUF_SIZE)) > 0) {
        for (ssize_t i = 0; i < numRead; ) {
            /* 偵測空洞區段 */
            if (buf[i] == 0) {
                ssize_t holeStart = i;

                while (i < numRead && buf[i] == 0)
                    i++;

                off_t holeSize = i - holeStart;

                if (lseek(outFd, holeSize, SEEK_CUR) == -1)
                    errExit("lseek");
            } else {
                /* 寫入非 0 資料 */
                ssize_t dataStart = i;

                while (i < numRead && buf[i] != 0)
                    i++;

                ssize_t dataSize = i - dataStart;

                if (write(outFd, &buf[dataStart], dataSize) != dataSize)
                    errExit("write");
            }
        }
    }
    if (numRead == -1)
        errExit("read");
}

/* ---------------------------------------------------------
 * 主程式：解析參數、開檔、呼叫複製函式
 * --------------------------------------------------------- */
int
main(int argc, char *argv[])
{
    int inFd, outFd;
    int mode;

    if (argc < 4)
        usageErr("%s file1 file2 mode[0=normal,1=sparse]\n", argv[0]);

    mode = atoi(argv[3]);

    /* 開啟來源檔案 */
    inFd = open(argv[1], O_RDONLY);
    if (inFd == -1)
        errExit("open source");

    /* 建立目的檔案 */
    outFd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC,
                 S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (outFd == -1)
        errExit("open dest");

    /* 根據模式選擇複製方式 */
    if (mode == 0)
        copy_normal(inFd, outFd);
    else
        copy_sparse(inFd, outFd);

    close(inFd);
    close(outFd);

    return 0;
}
