#include <unistd.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/types.h>

#define BUF_SIZE 1024

// struct linux_dirent64 並沒有在標準 C 標頭檔中定義，需要自行定義結構：
struct linux_dirent64 {
    ino_t        d_ino;    // inode number
    off_t        d_off;    // offset to next dirent
    unsigned short d_reclen; // length of this record
    unsigned char  d_type;   // file type
    char           d_name[]; // filename
};

int main() {
    int fd = open(".", O_RDONLY | O_DIRECTORY);
    char buf[BUF_SIZE];
    int nread;

    while ((nread = syscall(SYS_getdents64, fd, buf, BUF_SIZE)) > 0) {
//        struct linux_dirent *d;
        struct linux_dirent64 *d;
        int bpos;
        for (bpos = 0; bpos < nread;) {
//            d = (struct linux_dirent *) (buf + bpos);
            d = (struct linux_dirent64 *) (buf + bpos);
            printf("%s\n", d->d_name);
            bpos += d->d_reclen;
        }
    }
    close(fd);
    return 0;
}