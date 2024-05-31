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
template<typename T>
__device__ void setValueRandomly(float lower_limit, float upper_limit, T &valuetoBeSet,curandState *state){
    if(lower_limit < 0) lower_limit = fabs(lower_limit);
        valuetoBeSet = (T)curand_uniform(state) * upper_limit - lower_limit;
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
    __device__ void initialize(curandState *state){

        setValueRandomly(-20,20,position,state);
        setValueRandomly(-1,1,velocity,state);
        setValueRandomly(0.00001,1,frequency,state);
        loudness = 2;
        pulse_rate = 0.2;
        initial_pulse_rate = pulse_rate;
        personal_best_fitness=-FLT_MAX;
        personal_best_position=position;
        evaluateFitness();
    }

    __device__ void evaluateFitness(){
        fitness = ((float)ALPHA*k*position)/((float)1+(float)BETA*c*powf(position,2)) + ((float)GAMMA*r*pow(velocity,2));
        if(fitness > personal_best_fitness){
            personal_best_fitness = fitness;
            personal_best_position = position;
        }
    }
};