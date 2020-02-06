#include "ccl.h"
#include "sycl_base.hpp"

int main(int argc, char **argv)
{
    int i = 0;
    int retval = 0;
    size_t size = 0;
    size_t rank = 0;

    cl::sycl::queue q;
    cl::sycl::buffer<int, 1> sendbuf(COUNT);
    cl::sycl::buffer<int, 1> recvbuf(COUNT);

    ccl_request_t request;
    ccl_stream_t stream;

    ccl_init();
    ccl_get_comm_rank(NULL, &rank);
    ccl_get_comm_size(NULL, &size);
    
    if (create_sycl_queue(argc, argv, q) != 0) {
        return -1;
    }
    /* create SYCL stream */
    ccl_stream_create(ccl_stream_sycl, &q, &stream);

    {
        /* open sendbuf and recvbuf and initialize them on the CPU side */
        auto host_acc_sbuf = sendbuf.get_access<mode::write>();
        auto host_acc_rbuf = recvbuf.get_access<mode::write>();

        for (i = 0; i < COUNT; i++) {
            host_acc_sbuf[i] = rank;
            host_acc_rbuf[i] = 0;
        }
    }

    /* open sendbuf and modify it on the target device side */
    q.submit([&](handler& cgh) {
        auto dev_acc_sbuf = sendbuf.get_access<mode::write>(cgh);
        cgh.parallel_for<class allreduce_test_sbuf_modify>(range<1>{COUNT}, [=](item<1> id) {
            dev_acc_sbuf[id] += 1;
        });
    });
    /* exception handling */
    try {
        q.wait_and_throw();
    } catch (cl::sycl::exception const& e) {
        std::cout << "Caught synchronous SYCL exception:\n"
          << e.what() << std::endl;
    }

    /* invoke ccl_reduce on the CPU side */
    ccl_reduce(&sendbuf,
               &recvbuf,
               COUNT,
               ccl_dtype_int,
               ccl_reduction_sum,
               COLL_ROOT,
               NULL, /* attr */
               NULL, /* comm */
               stream,
               &request);

    ccl_wait(request);

    /* open recvbuf and check its correctness on the target device side */
    q.submit([&](handler& cgh) {
        auto dev_acc_rbuf = recvbuf.get_access<mode::write>(cgh);
        cgh.parallel_for<class allreduce_test_rbuf_check>(range<1>{COUNT}, [=](item<1> id) {
            if (rank == COLL_ROOT) {
                if (dev_acc_rbuf[id] != size * (size + 1) / 2) {
                    dev_acc_rbuf[id] = -1;
                }
            } else {
                if (dev_acc_rbuf[id] != 0) {
                    dev_acc_rbuf[id] = -1;
                }
            }
        });
    });
    /* exception handling */
    try {
        q.wait_and_throw();
    } catch (cl::sycl::exception const& e) {
        std::cout << "Caught synchronous SYCL exception:\n"
          << e.what() << std::endl;
    }

    /* print out the result of the test on the CPU side */
    auto host_acc_rbuf_new = recvbuf.get_access<mode::read>();
    if (rank == COLL_ROOT) {
        for (i = 0; i < COUNT; i++) {
            if (host_acc_rbuf_new[i] == -1) {
                cout << "FAILED for rank: " << rank << std::endl;
		retval = -1;
                break;
            }
        }
        if (i == COUNT) {
            cout << "PASSED" << std::endl;
        }
    }
    ccl_stream_free(stream);

    ccl_finalize();

    return retval;
}

