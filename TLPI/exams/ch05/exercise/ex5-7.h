#ifndef EX5_7_H
#define EX5_7_H

// 1. 引入必要的標頭檔
#include <sys/types.h> // 提供 ssize_t 形態
#include <sys/uio.h>   // 提供 struct iovec 結構定義

// 2. 支援 C++ 混用 (C/C++ Name Mangling 相容性)
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 模擬系統呼叫 readv() 的分散讀取函式
 * 
 * @param fd      檔案描述子 (File Descriptor)
 * @param iov     指向 struct iovec 陣列的指標
 * @param iovcnt  iovec 陣列的長度
 * @return ssize_t 成功回傳讀取總位元組數，EOF 回傳 0，失敗回傳 -1 並設定 errno
 */
ssize_t my_readv(int fd, const struct iovec *iov, int iovcnt);

/**
 * @brief 模擬系統呼叫 writev() 的聚集寫入函式
 */
ssize_t my_writev(int fd, const struct iovec *iov, int iovcnt);

#ifdef __cplusplus
}
#endif

#endif /* MY_UIO_H */