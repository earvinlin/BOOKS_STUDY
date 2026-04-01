// Listing 29-2: Creating a thread with the detached attribute

pthread_t thr;
pthread_attr_t attr;
int s;
s = pthread_attr_init(&attr); /* Assigns default values */
if (s != 0)
    errExitEN(s, "pthread_attr_init");

s = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
if (s != 0)
    errExitEN(s, "pthread_attr_setdetachstate");

s = pthread_create(&thr, &attr, threadFunc, (void *) 1);
if (s != 0)
    errExitEN(s, "pthread_create");

s = pthread_attr_destroy(&attr); /* No longer needed */
if (s != 0)
    errExitEN(s, "pthread_attr_destroy");
    