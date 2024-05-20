#include"dependencies_parallel/libs.h"
#include"dependencies_parallel/bat.h"

__global__
void init(Bat *bats,int N,unsigned long long seed){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(;i<N;i+=stride){
    bats[i].initialize(seed);
}

}

int main(){
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    size_t num_of_blocks=1,threads_per_block=4,N=1000;
    
    Bat *bats;     
    size_t size = sizeof(Bat) * N;
    cudaMallocManaged(&bats,size);
    init<<<num_of_blocks,threads_per_block>>>(bats,N,rand());       
    cudaDeviceSynchronize();  // Wait for GPU to finish before exiting
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);
    std::cout << "Time taken by sequential function: " 
              << duration.count() << " nanoseconds" << std::endl;

    cudaFree(bats);  // Free allocated memory

    return 0;
}