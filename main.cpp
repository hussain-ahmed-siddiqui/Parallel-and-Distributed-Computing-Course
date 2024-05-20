#include"dependencies/libs.h"
#include"dependencies/bat.h"

float global_best_fitness=-FLT_MIN;
float average_best_position_of_batSwarm;
float global_best_position=-FLT_MIN;
void updateBatFrequency(float &freq){
    freq = getRandomNumber(0,1); //simplified from fmin + (fmax - fmin) * randomNumber
    if(freq > 0.1 && getRandomNumber(0,1) > 0.5){
        freq*=getRandomNumber(0,1);
    }
}

void updateBatVelocity(float &current_v, float &current_p, float &freq, float lower_limit, float upper_limit){
    current_v = current_v + (current_p - global_best_position) * freq;
    if(current_v > upper_limit) current_v = upper_limit;
    else if(current_v < lower_limit) current_v = lower_limit;
}

void updateBatPulseRate(float &current_pr, float &initial_pr){
    current_pr = initial_pr* (1 - exp(-PULSE_RATE_CONSTANT * GAMMA));
}

void updateBatPosition(float &current_p, float &new_v, float &current_pr,float &initial_pr, float lower_limit, float upper_limit){
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

void updateBatLoudness(float &current_l){
    current_l *= LOUDNESS_CONSTANT;
}


int avg_personal_best_position_improv_counter=0;
bool ApplyStoppingCriteria(float &prev_avg, float new_avg){
    avg_personal_best_position_improv_counter++;
    if(fabs(new_avg) > fabs(prev_avg)){avg_personal_best_position_improv_counter=0;}
    if(avg_personal_best_position_improv_counter > 6) return true;
    return false;

}

float CalculateFitnessAverage(vector<Bat> &batSwarm){
        float sum = 0;
        for(Bat bat: batSwarm){
            sum += bat.personal_best_position;
        }
        average_best_position_of_batSwarm = sum/batSwarm.size();
        return average_best_position_of_batSwarm;  
    }

void setGlobalandAverage(vector<Bat> &batSwarm){
    float sum = 0;
    for(Bat bat: batSwarm){
        if(global_best_fitness < bat.fitness){
            global_best_fitness = bat.fitness;
            global_best_position = bat.personal_best_position;
        }
        sum+=bat.personal_best_position;
    }
    sum /= batSwarm.size();
    }

void startAlgo(vector<Bat> bats){
    setGlobalandAverage(bats);
    while(true){
        for(int i=0;i<bats.size();i++){
            updateBatFrequency(bats[i].frequency);
            updateBatVelocity(bats[i].velocity,bats[i].position, bats[i].frequency,VELOCITY_LOWER_BOUND,VELOCITY_UPPER_BOUND);
            updateBatPosition(bats[i].position,bats[i].velocity,bats[i].pulse_rate,bats[i].initial_pulse_rate,POSITION_LOWER_BOUND,POSITION_UPPER_BOUND);
            updateBatLoudness(bats[i].loudness);
            updateBatPulseRate(bats[i].pulse_rate,bats[i].initial_pulse_rate);
            bats[i].evaluateFitness();
            if(bats[i].fitness > global_best_fitness){
                global_best_fitness = bats[i].fitness;
                global_best_position = bats[i].position;
            
            }
            
        }
        float prev_avg = average_best_position_of_batSwarm;
    if(ApplyStoppingCriteria(prev_avg,CalculateFitnessAverage(bats))) break;
    cout<<"Global fitness: "<<global_best_fitness<<" Average position: " <<average_best_position_of_batSwarm<<"\n";
    
    }


}

int main(){
    vector<Bat> bats(50);
    startAlgo(bats);
}