#include <stdio.h>
#include <stdlib.h>
#include <sys/utsname.h>

/*
struct utsname {
    char sysname[];    // 作業系統名稱（例如：Linux、Darwin）
    char nodename[];   // 網路主機名稱（Node/Host name）
    char release[];    // 系統核心發行版本（Kernel release，例如：5.15.0-88-generic）
    char version[];    // 系統核心編譯版本號與建置時間（Kernel version）
    char machine[];    // 硬體架構名稱（例如：x86_64、aarch64）
#ifdef _GNU_SOURCE
    char domainname[]; // NIS/YP 網域名稱（GNU 延伸，非所有 POSIX 系統皆有）
#endif
*/


int main(void) {
    struct utsname buffer;

    if (uname(&buffer) != 0) {
        perror("uname 呼叫失敗");
        return EXIT_FAILURE;
    }

    printf("作業系統名稱 (sysname)   : %s\n", buffer.sysname);
    printf("主機名稱     (nodename)  : %s\n", buffer.nodename);
    printf("核心發行版本 (release)   : %s\n", buffer.release);
    printf("核心詳細版本 (version)   : %s\n", buffer.version);
    printf("硬體架構     (machine)   : %s\n", buffer.machine);
    /* 檢查是否具備 domainname 成員（GNU 擴充） */
#if defined(_GNU_SOURCE) || defined(__USE_GNU)
    printf("YP網域名稱   (domainname): %s\n", buffer.domainname);
#else
    printf("YP網域名稱   (domainname): [不支援]\n");
#endif
    return EXIT_SUCCESS;
}

