
#define ALPHA 1
#define BETA 1
#define GAMMA 1
#define k 1
#define c 1
#define r 1
#define VELOCITY_UPPER_BOUND -1
#define VELOCITY_LOWER_BOUND  1
#define POSITION_UPPER_BOUND -10
#define POSITION_LOWER_BOUND  10

using namespace std;

   auto now = chrono::high_resolution_clock::now();

    // Convert time point to a number
    auto time_count = now.time_since_epoch().count();

    // Get the thread ID
    auto thread_id = this_thread::get_id();

    // Hash the thread ID
    size_t thread_id_hash = hash<thread::id>{}(thread_id);

    // Combine the high-resolution time with the thread ID hash
    unsigned seed = static_cast<unsigned>(time_count) ^ static_cast<unsigned>(thread_id_hash);

    mt19937 gen(seed); // Seed the generator with current time

    double getRandomNumber(int lower_limit, int upper_limit){
        uniform_real_distribution<> randum(lower_limit, upper_limit); // define the range
        return randum(gen);
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

    Bat(){
        position = getRandomNumber(-10,10);
        velocity = getRandomNumber(-1,1);
        frequency = getRandomNumber(0,1);
        loudness = 2;
        pulse_rate = 0.1;
    }

    void evaluateFitness(){
        fitness = (ALPHA*k*position)/(1+BETA*c*pow(position,2)) + (GAMMA*r*pow(velocity,2));
        if(fitness > personal_best_fitness){
            personal_best_fitness = fitness;
            personal_best_position = position;
        }
    }
};