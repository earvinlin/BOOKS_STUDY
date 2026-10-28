#include <stdio.h>
#include <stdbool.h>

#define N 5
#define M 4

void calculateNeed(
    int Max[N][M],
    int Allocation[N][M],
    int Need[N][M])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            Need[i][j] =
                Max[i][j] - Allocation[i][j];
        }
    }
}

bool isSafe(int Available[M], int Allocation[N][M], int Need[N][M])
{
    int Work[M];
    bool Finish[N];

    /* Work = Available */
    for (int i = 0; i < M; i++)
        Work[i] = Available[i];

    for (int i = 0; i < N; i++)
        Finish[i] = false;

    int safeSequence[N];
    int count = 0;

    while (count < N)
    {
        bool found = false;

        for (int i = 0; i < N; i++)
        {
            if (Finish[i] == false)
            {
                int j;

                for (j = 0; j < M; j++)
                {
                    if (Need[i][j] > Work[j])
                        break;
                }

                if (j == M)
                {
                    for (int k = 0; k < M; k++)
                    {
                        Work[k] += Allocation[i][k];
                    }

                    safeSequence[count++] = i;
                    Finish[i] = true;
                    found = true;
                }
            }
        }

        if (found == false)
        {
            printf("System is NOT safe\n");
            return false;
        }
    }

    printf("System is SAFE\n");

    printf("Safe Sequence: ");

    for (int i = 0; i < N; i++)
    {
        printf("P%d", safeSequence[i]);

        if (i != N - 1)
            printf(" -> ");
    }

    printf("\n");

    return true;
}

int main()
{
    int Allocation[N][M] = {
        {0, 0, 1, 2},
        {1, 0, 0, 0},
        {1, 3, 5, 4},
        {0, 6, 3, 2},
        {0, 0, 1, 4}};

    int Max[N][M] = {
        {0, 0, 2, 4},
        {2, 7, 5, 0},
        {3, 6, 10, 10},
        {0, 6, 8, 4},
        {0, 6, 6, 10}};

    int Available[M] = {1, 5, 2, 0};

    int Need[N][M];

    calculateNeed(Max, Allocation, Need);

    printf("Need Matrix\n");

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            printf("%3d", Need[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    isSafe(Available, Allocation, Need);

    return 0;
}
