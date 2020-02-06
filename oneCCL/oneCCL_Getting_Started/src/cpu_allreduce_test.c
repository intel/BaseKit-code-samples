#include <stdio.h>
#include "ccl.h"

#define COUNT 128

int main(int argc, char** argv)
{
    int i = 0;
    size_t size = 0;
    size_t rank = 0;

    int sendbuf[COUNT];
    int recvbuf[COUNT];

    ccl_request_t request;
    ccl_stream_t stream;

    ccl_init();

    ccl_get_comm_rank(NULL, &rank);
    ccl_get_comm_size(NULL, &size);

    /* create CPU stream */
    ccl_stream_create(ccl_stream_cpu, NULL, &stream);

    /* initialize sendbuf */
    for (i = 0; i < COUNT; i++) {
        sendbuf[i] = rank;
    }

    /* modify sendbuf */
    for (i = 0; i < COUNT; i++) {
        sendbuf[i] += 1;
    }

    /* invoke ccl_allreduce */
    ccl_allreduce(sendbuf,
                  recvbuf,
                  COUNT,
                  ccl_dtype_int,
                  ccl_reduction_sum,
                  NULL, /* attr */
                  NULL, /* comm */
                  stream,
                  &request);

    ccl_wait(request);

    /* check correctness of recvbuf */
    for (i = 0; i < COUNT; i++) {
       if (recvbuf[i] != size * (size + 1) / 2) {
           recvbuf[i] = -1;
       }
    }

    /* print out the result of the test */
    if (rank == 0) {
        for (i = 0; i < COUNT; i++) {
            if (recvbuf[i] == -1) {
                printf("FAILED\n");
                break;
            }
        }
        if (i == COUNT) {
            printf("PASSED\n");
        }
    }

    ccl_stream_free(stream);

    ccl_finalize();

    return 0;
}
