#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/wait.h>

void error(char *msg)
{
    fprintf(stderr, "%s: %s\n", msg, strerror(errno));
    exit(1);
}

int main(int argc, char *argv[])
{
    char *phrase = argv[1];
    char *vars[] = {"RSS_FEED=http://www.cnn.com/rss/celebs.xml", NULL};

    FILE *f = fopen("stories.txt", "w");
    if (!f) {
        error("Can't open stories.txt");
    }

    pid_t pid = fork();
    if (pid == -1) {
        error("Can't fork process");
    }
    if (!pid) {
        if (dup2(fileno(f), 1) == -1) {
            error("Can't redirect Standard Output");
        }
        if (execle("/usr/bin/python", "/usr/bin/python", "./rssgossip.py",
                    phrase, NULL, vars) == -1) {
            error("Can't run script");
        }
    }

    int pid_status;
    /**
     * waitpid(pid, pid_status, options)
     * pid          當父行程分叉出子行程時被供的行程id
     * pid_status   儲存該行程的退出資訊，waitpid()會更新它，必須是指標
     * options      ref. docs。若選項設為0，該函式就會等到該行程結束
     */
    if (waitpid(pid, &pid_status, 0) == -1) {
        error("waiting child process is error");
    }
    /**
     * WEXITSTATUS為巨集
     * pid_status包含一些資訊片段，且只有前8個位元代表退出狀態，該巨集只告訴你那8個
     * 位元的值
     */
    if (WEXITSTATUS(pid_status)) {
        puts("Error status non-zero");
    }

    return 0;
}

