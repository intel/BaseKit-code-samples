#include <iostream>
#include <stdio.h>
#include <CL/sycl.hpp>
#include "ccl.hpp"


#define COUNT     (10 * 1024 * 1024)
#define COLL_ROOT (0)

using namespace std;
using namespace cl::sycl;
using namespace cl::sycl::access;
inline bool has_gpu()
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    for (const auto& device : devices)
    {
        if (device.is_gpu())
        {
            return true;
        }
    }
    return false;
}
inline bool has_accelerator()
{
    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    for (const auto& device : devices)
    {
        if (device.is_accelerator())
        {
            return true;
        }
    }
    return false;
}
inline int create_sycl_queue(int argc, char **argv, cl::sycl::queue &queue)
{
   auto exception_handler = [&](cl::sycl::exception_list elist) {
      for (std::exception_ptr const& e : elist) {
        try {
          std::rethrow_exception(e);
        } catch (cl::sycl::exception const& e) {
          std::cout << "failure" << std::endl;
          std::terminate();
        }
      }
    };

    std::unique_ptr<cl::sycl::device_selector> selector;
    if (argc == 2)
    {
        if (strcmp(argv[1], "cpu") == 0)
        {
            selector.reset(new cl::sycl::cpu_selector());
        }
        else if (strcmp(argv[1], "gpu") == 0)
        {
            if (has_gpu()) 
            {
                selector.reset(new cl::sycl::gpu_selector());
            }
            else if (has_accelerator()) 
            {
                selector.reset(new cl::sycl::host_selector());
                std::cout << "Accelerator is unavailable for multiprocessing, host_selector has been created instead of default_selector." << std::endl;
            }
            else
            {
                selector.reset(new cl::sycl::default_selector());
                std::cout << "GPU is unavailable, default_selector has been created instead of gpu_selector." << std::endl;
            }
        }
        else if (strcmp(argv[1], "host") == 0)
        {
            selector.reset(new cl::sycl::host_selector());
        }
        else if (strcmp(argv[1], "default") == 0)
        {
            if (!has_accelerator())
            {
                selector.reset(new cl::sycl::default_selector());
            }
            else
            {
                selector.reset(new cl::sycl::host_selector());
                std::cout << "Accelerator is unavailable for multiprocessing, host_selector has been created instead of default_selector." << std::endl;
            }
        }
        else
        {
            std::cerr << "Please provide device type: cpu | gpu | host | default " << std::endl;
            return -1;
        }
        queue = cl::sycl::queue(*selector, exception_handler);
        std::cout << "Provided device type " << argv[1] << "\nRunning on "
                  << queue.get_device().get_info<cl::sycl::info::device::name>()
                  << "\n";
    }
    else
    {
        std::cerr << "Please provide device type: cpu | gpu | host | default " << std::endl;
        return -1;
    }
    return 0;
}
