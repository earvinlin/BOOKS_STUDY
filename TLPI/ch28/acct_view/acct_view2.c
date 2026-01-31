#define _GNU_SOURCE
#include <fcntl.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/acct.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define TIME_BUF_SIZE 100

/* Convert comp_t to long long */
static long long comptToLL(comp_t ct) {
    int exp = (ct >> 13) & 7;      /* 3-bit exponent */
    int mant = ct & 0x1FFF;       /* 13-bit mantissa */
    return (long long)mant << (exp * 3);
}

int main(int argc, char *argv[])
{
    int fd;
    struct acct_v3 ac;
    ssize_t n;
    char timeBuf[TIME_BUF_SIZE];
    struct tm *loc;
    time_t t;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s pacct_file\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    fd = open(argv[1], O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    printf("command     flags exit user   start time           CPU   elapsed\n");

    while ((n = read(fd, &ac, sizeof(ac))) == sizeof(ac)) {

        /* command */
        printf("%-10.10s ", ac.ac_comm);

        /* flags */
        printf("%c", (ac.ac_flag & AFORK) ? 'F' : '-');
        printf("%c", (ac.ac_flag & ASU)   ? 'S' : '-');
        printf("%c", (ac.ac_flag & AXSIG) ? 'X' : '-');
        printf("%c ", (ac.ac_flag & ACORE)? 'C' : '-');

        /* exit code */
        printf("%5u ", ac.ac_exitcode);

        /* user */
        printf("%5u ", ac.ac_uid);

        /* start time */
        t = ac.ac_btime;
        loc = localtime(&t);
        if (loc) {
            strftime(timeBuf, TIME_BUF_SIZE, "%Y-%m-%d %H:%M:%S", loc);
            printf("%s ", timeBuf);
        } else {
            printf("??? ");
        }

        /* CPU time */
        double cpu = (comptToLL(ac.ac_utime) + comptToLL(ac.ac_stime)) /
                     sysconf(_SC_CLK_TCK);

        /* elapsed time */
        double elapsed = comptToLL(ac.ac_etime) / sysconf(_SC_CLK_TCK);

        printf("%6.2f %7.2f\n", cpu, elapsed);
    }

    close(fd);
    return 0;
}
