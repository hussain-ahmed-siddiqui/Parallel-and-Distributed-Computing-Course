#include"dependencies_parallel/libs.h"
#include"dependencies_parallel/bat.h"

__global__
void init(Bat *bats,int N,float seed){
    curandState state;

    // Initialize curand state with a unique sequence number
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(;i<N;i+=stride){
    curand_init(seed, i, 0, &state); 

    bats[i].initialize(&state);
}

}

float global_best_fitness=-FLT_MIN;
float average_best_position_of_batSwarm;
float global_best_position=-FLT_MIN;
__global__ void updateBatFrequency(float &freq, curandState *state){
    setValueRandomly(0,1,freq,state); //simplified from fmin + (fmax - fmin) * randomNumber
    if(freq > 0.1 && curand_uniform(state) > 0.5){
        freq*=curand_uniform(state);
    }
}

__global__ void updateBatVelocity(float &current_v, float &current_p, float &freq, float lower_limit, float upper_limit){
    current_v = current_v + (current_p - global_best_position) * freq;
    if(current_v > upper_limit) current_v = upper_limit;
    else if(current_v < lower_limit) current_v = lower_limit;
}

__global__ void updateBatPulseRate(float &current_pr, float &initial_pr){
    current_pr = initial_pr* (1 - exp(-PULSE_RATE_CONSTANT * GAMMA));
}

__global__ void updateBatPosition(float &current_p, float &new_v, float &current_pr,float &initial_pr, float lower_limit, float upper_limit){
    float prev_pr = current_pr;
    updateBatPulseRate(current_pr, initial_pr);
    if (current_pr < prev_pr ) {  // Assuming rand() returns a float between 0 and 1
        // Perform a local search towards the global best position
        current_p += (global_best_position - current_p) * 0.01; // Example: move halfway towards global best
    } else {
        // Regular position update
        current_p += new_v;
    }
    if(current_p > upper_limit) current_p = upper_limit;
    else if(current_p < lower_limit) current_p = lower_limit;
}

__global__ void updateBatLoudness(float &current_l){
    current_l *= LOUDNESS_CONSTANT;
}

int main(){
    
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    size_t num_of_blocks=1,threads_per_block=2,N=100;
    
    Bat *bats;     
    size_t size = sizeof(Bat) * N;
    
    cudaMallocManaged(&bats,size);
    init<<<num_of_blocks,threads_per_block>>>(bats,N,rand());       
    cudaDeviceSynchronize();  // Wait for GPU to finish before exiting
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);
    std::cout << "Time taken by sequential function: " 
              << duration.count() << " nanoseconds" << std::endl;
    for(int i=0;i<N;i++){
        printf("v: %f, p: %f, f: %f, l: %f, pr: %f, fitness: %f\n",bats[i].velocity,bats[i].position,bats[i].frequency,bats[i].loudness,bats[i].pulse_rate,bats[i].fitness);
    }
    cudaFree(bats);  // Free allocated memory
    
    
    return 0;
}