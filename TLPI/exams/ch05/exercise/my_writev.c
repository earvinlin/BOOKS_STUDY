/**
 * ex5-7 use module
 * 將多個 iovec 的資料複製拼接到一個由 malloc() 分配的大型動態緩衝區中，再呼叫一
 * 次 write() 將其寫出。
 * 
 * 在產生目的檔時，通常會加上以下推薦的旗標以提升程式碼品質與利於偵錯：
 * gcc -Wall -Wextra -g -c my_writev.c -o my_writev.o
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

ssize_t my_writev(int fd, const struct iovec *iov, int iovcnt) {
    if (iovcnt <= 0) {
        errno = EINVAL;
        return -1;
    }

    // 1. 計算所有 iovec 緩衝區的總位元組數
    size_t total_bytes = 0;
    for (int i = 0; i < iovcnt; i++) {
        total_bytes += iov[i].iov_len;
    }

    if (total_bytes == 0) return 0;

    // 2. 使用 malloc 分配暫存記憶體
    char *buf = (char *)malloc(total_bytes);
    if (buf == NULL) {
        return -1; // ENOMEM
    }

    // 3. 將各個 iovec 中的資料「聚集 (Gather)」複製到連續的暫存區
    size_t offset = 0;
    for (int i = 0; i < iovcnt; i++) {
        if (iov[i].iov_len > 0 && iov[i].iov_base != NULL) {
            memcpy(buf + offset, iov[i].iov_base, iov[i].iov_len);
            offset += iov[i].iov_len;
        }
    }

    // 4. 執行實際的 write() 系統呼叫
    ssize_t bytes_written = write(fd, buf, total_bytes);

    // 5. 釋放 malloc 分配的記憶體
    free(buf);

    return bytes_written;
}