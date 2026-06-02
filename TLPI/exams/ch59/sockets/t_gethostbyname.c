#define _BSD_SOURCE /* To get hstrerror() declaration from <netdb.h> */
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#if defined(USE_MYLIB_INTEL)
    #include "../../../tlpi-book/mylib-intel/tlpi_hdr.h"   // For linux(intel) use
#else
    #include "../../../tlpi-book/mylib/tlpi_hdr.h"         // For macnb's vmubuntu(arm) use
#endif

int main(int argc, char *argv[])
{
    struct hostent *h;
    char **pp;
    char str[INET6_ADDRSTRLEN];

    for (argv++; *argv != NULL; argv++) {
        h = gethostbyname(*argv);
        if (h == NULL) {
            fprintf(stderr, "gethostbyname() failed for '%s': %s\n", *argv, hstrerror(h_errno));
            continue;
        }

        printf("Canonical name: %s\n", h->h_name);
        printf(" alias(es): ");

        for (pp = h->h_aliases; *pp != NULL; pp++)
            printf(" %s", *pp);

        printf("\n");
        printf(" address type: %s\n",
            (h->h_addrtype == AF_INET) ? "AF_INET" :
            (h->h_addrtype == AF_INET6) ? "AF_INET6" : "???");

        if (h->h_addrtype == AF_INET || h->h_addrtype == AF_INET6) {
            printf(" address(es): ");
            for (pp = h->h_addr_list; *pp != NULL; pp++)
                printf(" %s", inet_ntop(h->h_addrtype, *pp, str, INET6_ADDRSTRLEN));
            printf("\n");
        }
    }
    exit(EXIT_SUCCESS);
}
/*
// 現代化 getaddrinfo() 版本（完整可編譯）
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <hostname>...\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < argc; i++) {
        const char *host = argv[i];
        struct addrinfo hints;
        struct addrinfo *result, *rp;
        char addrstr[INET6_ADDRSTRLEN];

        memset(&hints, 0, sizeof(struct addrinfo));
        hints.ai_family = AF_UNSPEC;      // IPv4 + IPv6
        hints.ai_socktype = SOCK_STREAM;  // 任意即可
        hints.ai_flags = AI_CANONNAME;    // 要求 canonical name

        int s = getaddrinfo(host, NULL, &hints, &result);
        if (s != 0) {
            fprintf(stderr, "getaddrinfo() failed for '%s': %s\n",
                    host, gai_strerror(s));
            continue;
        }

        printf("Host: %s\n", host);

        if (result->ai_canonname)
            printf(" Canonical name: %s\n", result->ai_canonname);

        printf(" Address(es):");

        for (rp = result; rp != NULL; rp = rp->ai_next) {
            void *addr;
            const char *type;

            if (rp->ai_family == AF_INET) {
                struct sockaddr_in *ipv4 = (struct sockaddr_in *)rp->ai_addr;
                addr = &(ipv4->sin_addr);
                type = "IPv4";
            } else if (rp->ai_family == AF_INET6) {
                struct sockaddr_in6 *ipv6 = (struct sockaddr_in6 *)rp->ai_addr;
                addr = &(ipv6->sin6_addr);
                type = "IPv6";
            } else {
                continue;
            }

            inet_ntop(rp->ai_family, addr, addrstr, sizeof(addrstr));
            printf(" [%s] %s", type, addrstr);
        }

        printf("\n\n");
        freeaddrinfo(result);
    }

    exit(EXIT_SUCCESS);
}

*/
