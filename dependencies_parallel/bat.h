// #include"libs.h"
#include <curand_kernel.h>
#define ALPHA 1.5
#define BETA 2
#define GAMMA 1.5
#define k 1
#define c 0.01
#define r 0.05
#define VELOCITY_UPPER_BOUND   10
#define VELOCITY_LOWER_BOUND  -5
#define POSITION_UPPER_BOUND   20
#define POSITION_LOWER_BOUND  -200
#define LOUDNESS_CONSTANT 0.95
#define PULSE_RATE_CONSTANT 0.1
using namespace std;

__device__ unsigned int getRandomNumber(unsigned int &seed) {
    const unsigned int a = 1664525;
    const unsigned int b = 1013904223;
    seed = a * seed + b;
    return seed & 0x7FFFFFFF;
}

struct Bat{
    float position;
    float velocity;
    float frequency;
    float pulse_rate;
    float loudness;
    float fitness;
    float personal_best_fitness;
    float personal_best_position;
    float initial_pulse_rate;
    __device__ void initialize(unsigned long long seed){
         int id = threadIdx.x + blockIdx.x * blockDim.x;
                unsigned int localSeed =  seed + id;  // Different seed for each thread
        position = (getRandomNumber(localSeed) % 20000) / 1000.0f - 10.0f;
        velocity = (getRandomNumber(localSeed) % 2000) / 1000.0f - 1.0f;
        frequency =  (getRandomNumber(localSeed) % 1000) / 1000.0f;  
        loudness = 2;
        pulse_rate = 0.2;
        initial_pulse_rate = pulse_rate;
    }

    __device__ void evaluateFitness(){
        fitness = ((float)ALPHA*k*position)/((float)1+(float)BETA*c*powf(position,2)) + ((float)GAMMA*r*pow(velocity,2));
        if(fitness > personal_best_fitness){
            personal_best_fitness = fitness;
            personal_best_position = position;
        }
    }
};