/**
 * ex5-7 use module
 * 先分配一個足以容納總讀取量的動態緩衝區，呼叫一次 read() 將資料讀入，再依序「分散」
 * 複製回各個 iovec 緩衝區中。
 *
 * 在產生目的檔時，通常會加上以下推薦的旗標以提升程式碼品質與利於偵錯：
 * gcc -Wall -Wextra -g -c my_readv.c -o my_readv.o
 *  • -Wall -Wextra：開啟大部分的編譯警告訊息，協助找出潛在 Bug。
 *  • -g：加入偵錯資訊（Debug Symbols），便於後續使用 GDB 等工具偵錯。
 *  • -I<dir>：如果程式碼中用到了非預設路徑的標頭檔（.h），可用 -I 指定搜尋目錄
 *            （例如：gcc -I./include -c main.c）。
 */
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

ssize_t my_readv(int fd, const struct iovec *iov, int iovcnt) {
    if (iovcnt <= 0) {
        errno = EINVAL;
        return -1;
    }

    // 1. 計算欲讀取的總位元組數
    size_t total_bytes = 0;
    for (int i = 0; i < iovcnt; i++) {
        total_bytes += iov[i].iov_len;
    }

    if (total_bytes == 0) return 0;

    // 2. 使用 malloc 分配暫存記憶體
    char *buf = (char *)malloc(total_bytes);
    if (buf == NULL) {
        return -1;
    }

    // 3. 執行單次 read() 讀取資料
    ssize_t bytes_read = read(fd, buf, total_bytes);
    if (bytes_read <= 0) {
        free(buf);
        return bytes_read; // 回傳 0 (EOF) 或 -1 (Error)
    }

    // 4. 將讀取到的資料「分散 (Scatter)」填回各個 iovec 緩衝區
    size_t offset = 0;
    size_t bytes_remaining = bytes_read;

    for (int i = 0; i < iovcnt && bytes_remaining > 0; i++) {
        // 計算本次要複製進 iov[i] 的位元組數 (不能超過該 iov 的容量與剩餘讀取量)
        size_t copy_len = iov[i].iov_len;
        if (copy_len > bytes_remaining) {
            copy_len = bytes_remaining;
        }

        if (copy_len > 0 && iov[i].iov_base != NULL) {
            memcpy(iov[i].iov_base, buf + offset, copy_len);
            offset += copy_len;
            bytes_remaining -= copy_len;
        }
    }

    // 5. 釋放暫存區
    free(buf);

    return bytes_read;
}