#include <iostream>

void test_hardware_detection();
void test_kernel_tail_handling();
void test_static_arena();
void test_format_validation();
void test_runtime_engine();

int main() {
    std::cout << "==========================================================\n";
    std::cout << "       MuRDoK Inference Engine — Test Suite Runner        \n";
    std::cout << "==========================================================\n\n";

    test_hardware_detection();
    test_kernel_tail_handling();
    test_static_arena();
    test_format_validation();
    test_runtime_engine();

    std::cout << "==========================================================\n";
    std::cout << "             ALL TESTS PASSED SUCCESSFULLY!               \n";
    std::cout << "==========================================================\n";
    return 0;
}
