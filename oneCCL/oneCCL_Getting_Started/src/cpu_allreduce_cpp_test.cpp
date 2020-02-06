#include <iostream>
#include <stdio.h>
#include "ccl.hpp"

#define COUNT 128

using namespace std;

int main(int argc, char** argv)
{
    int i = 0;
    int size = 0;
    int rank = 0;

    auto sendbuf = new int[COUNT];
    auto recvbuf = new int[COUNT];

    auto comm = ccl::environment::instance().create_communicator();
    auto stream = ccl::environment::instance().create_stream();

    rank = comm->rank();
    size = comm->size();

    /* initialize sendbuf */
    for (i = 0; i < COUNT; i++) {
        sendbuf[i] = rank;
    }

    /* modify sendbuf */
    for (i = 0; i < COUNT; i++) {
        sendbuf[i] += 1;
    }

    /* invoke ccl_allreduce */
    comm->allreduce(sendbuf,
                   recvbuf,
                   COUNT,
                   ccl::reduction::sum,
                   nullptr, /* attr */
                   stream)->wait();

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
                cout << "FAILED" << endl;
                break;
            }
        }
        if (i == COUNT) {
            cout << "PASSED" << endl;
        }
    }

    return 0;
}
