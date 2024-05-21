#include"dependencies_parallel/libs.h"
#include"dependencies_parallel/bat.h"

__device__
void init(Bat *bats,int N,float seed){
 curandState *state = new curandState; 
    // Initialize curand state with a unique sequence number
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(;i<N;i+=stride){
    curand_init(seed, i, 0, state); 

    bats[i].initialize(state);

}
    delete state;  // Clean up state

}


__device__ void updateBatFrequency(float &freq,curandState *state){
    setValueRandomly(0,1,freq,state); //simplified from fmin + (fmax - fmin) * randomNumber
    if(freq > 0.1 && curand_uniform(state) > 0.5){
        freq*=curand_uniform(state);
    }
}

__device__ void updateBatVelocity(float &current_v, float &current_p, float &freq,float global_best_position, float lower_limit, float upper_limit){
    current_v = current_v + (current_p - global_best_position) * freq;
    if(current_v > upper_limit) current_v = upper_limit;
    else if(current_v < lower_limit) current_v = lower_limit;
}

__device__ void updateBatPulseRate(float &current_pr, float &initial_pr){
    current_pr = initial_pr* (1 - exp(-PULSE_RATE_CONSTANT * GAMMA));
}

__device__ void updateBatPosition(float &current_p, float &new_v, float &current_pr,float &initial_pr,float global_best_position, float lower_limit, float upper_limit){
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

__device__ void updateBatLoudness(float &current_l){
    current_l *= LOUDNESS_CONSTANT;
}

// __device__ volatile bool stopFlag = false;
__device__ void performWork(Bat *bat,float global_best_position,curandState *state){
    
    updateBatFrequency(bat->frequency,state);
    updateBatVelocity(bat->velocity,bat->position,bat->frequency,global_best_position,VELOCITY_LOWER_BOUND,VELOCITY_UPPER_BOUND);
    updateBatPosition(bat->position,bat->velocity,bat->pulse_rate,bat->initial_pulse_rate,global_best_position,POSITION_LOWER_BOUND,POSITION_UPPER_BOUND);
    updateBatLoudness(bat->loudness);
    updateBatPulseRate(bat->pulse_rate,bat->initial_pulse_rate);
    bat->evaluateFitness();
}


__device__ int avg_personal_best_position_improv_counter=0;
__device__ void ApplyStoppingCriteria(float &prev_avg, float new_avg,volatile bool &stopFlag){
    avg_personal_best_position_improv_counter++;
    if(fabs(new_avg) > fabs(prev_avg)){avg_personal_best_position_improv_counter=0;}
    if(avg_personal_best_position_improv_counter > 5) stopFlag = true;
        __threadfence();  // Ensure the write to 'flag' is visible to all threads across the device

    
}


__device__ void CalculateFitnessAverage(Bat *batSwarm,int N, float& average_best_position_of_batSwarm){
        float sum = 0;
        for(int i=0;i<N;i++){
            sum += batSwarm[i].personal_best_position;
        }
        average_best_position_of_batSwarm = sum/N;
        
    }


__device__ void setGlobalandAverage(Bat *batSwarm, int N, float& global_best_fitness, float& global_best_position, float &average_best_position_of_batSwarm){
    global_best_fitness=-FLT_MAX;
    global_best_position=FLT_MAX;
    float sum = 0;
    for(int i=0;i<N;i++){
        if(global_best_fitness < batSwarm[i].fitness){
            global_best_fitness = batSwarm[i].fitness;
            global_best_position = batSwarm[i].personal_best_position;
        }
        sum+=batSwarm[i].personal_best_position;
        // printf("sum: %f, personal best position: %f")
    }
    average_best_position_of_batSwarm = sum/(float)N;
    }


// __device__ void updateControlFlag() {
//     stopFlag = true;
//     __threadfence();  // Ensure the write to 'flag' is visible to all threads across the device
//     // Optionally follow with an atomic operation if needed to further signal or manage control
// }

__device__ void startAlgo(Bat *bats, int N, unsigned long long seed){
    __shared__ float global_best_fitness;
    __shared__ float average_best_position_of_batSwarm;
    __shared__ float global_best_position;
    __shared__ volatile bool stopFlag;
    curandState *state = new curandState;  // Should ideally be per-thread and persistent, not recreated in a loop

    // Initialize random states once per thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        curand_init(seed, idx, 0, state); 
    }
    setGlobalandAverage(bats, N, global_best_fitness, global_best_position, average_best_position_of_batSwarm);        

    while(!stopFlag){
        int stride =  (blockDim.x-1) * gridDim.x;
        for(int i= threadIdx.x + blockDim.x * blockIdx.x;i<N;i+=stride){
            if(threadIdx.x!=4) {performWork(&bats[i],global_best_position,state);
            printf("this is thread: %d and i= %d\t",threadIdx.x,i);
            }

            __syncthreads();
        }
        if(threadIdx.x==0){
            float prev_avg = average_best_position_of_batSwarm;
            CalculateFitnessAverage(bats, N, average_best_position_of_batSwarm);
            ApplyStoppingCriteria(prev_avg, average_best_position_of_batSwarm,stopFlag);
        }
            printf("\n");

    }

    delete state;  // Clean up the state
}


__global__ void launchGpuKernel(Bat *bats,int N, float seed){
    if(threadIdx.x==0) init(bats,N,seed);
    __syncthreads();
    
    startAlgo(bats,N,seed);
}


int main(){
    
    const size_t num_of_blocks=1,threads_per_block=5,N=100;
    
    Bat* bats;
    cudaMalloc(&bats, N * sizeof(Bat));    
    unsigned long long seed = time(NULL);  // Using time as seed for demonstration
    auto start = chrono::high_resolution_clock::now();
    launchGpuKernel<<<num_of_blocks,threads_per_block>>>(bats,N,seed);

    cudaDeviceSynchronize();
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
    std::cout << "Time taken by sequential function: " 
              << duration.count() << " nanoseconds" << std::endl;

    cudaFree(bats);
    return 0;
}