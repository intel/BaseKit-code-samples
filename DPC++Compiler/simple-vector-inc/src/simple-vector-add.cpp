#include <CL/sycl.hpp>
using namespace cl::sycl;
static const size_t N = 2;
int main() {
        default_selector my_selector;
        queue my_queue(my_selector);
        std::cout << "Device : " << my_queue.get_device().get_info<info::device::name>() << std::endl;
        int vector1[N] = {10,10};
        int vector2[N] = {20,20};
        std::cout << "Inputs vector1 : " << vector1[0] << ", " << vector1[1] << std::endl;
        std::cout << "Inputs vector2 : " << vector2[0] << ", " << vector2[1] << std::endl;

        buffer<int, 1> vector1_buffer(vector1, range<1>(N));
        buffer<int, 1> vector2_buffer(vector2, range<1>(N));

        my_queue.submit([&] (handler &my_handler){
                auto vector1_accessor = vector1_buffer.template get_access<access::mode::read_write>(my_handler);
                auto vector2_accessor = vector2_buffer.template get_access<access::mode::read>(my_handler);
                my_handler.parallel_for<class test>(range<1>(N), [=](id<1> index){
                    vector1_accessor[index] += vector2_accessor[index];
                });
        });

        my_queue.wait_and_throw();
        vector1_buffer.get_access<access::mode::read>();
        std::cout << "Output : " << vector1[0] << ", " << vector1[1] << std::endl;
}
