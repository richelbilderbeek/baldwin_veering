/**
 * @file rl_logic.hpp
 * @brief containing the reinforcement learning logic for the agents "brain"
 */

#ifndef RL_LOGIC
#define RL_LOGIC
#include <iostream>
#include "constants.h"

#include <fstream>
#include <string>
#include <stdexcept>
#include <functional>
#include <random>
#include <algorithm>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include "cereal/cereal.hpp"
#include "cereal/types/vector.hpp"
//#define DEBUG

namespace Joleste
{

#ifndef RNG
#define RNG
    extern std::mt19937 rng;
#endif

    class agent_brain
    {

    private:
        typedef double reward_type;
        // parameters
        unsigned int action_number;
        reward_type alpha;
        reward_type gamma;
        double epsilon;
        std::string initial_attitude;
        std::uniform_real_distribution<double> dist;
        std::uniform_int_distribution<int> int_dist;
        std::function <double()> prob01;
        std::function <int()> randInt;
        unsigned int current_choice;
        unsigned int future_choice;
        reward_type current_reward;
        long int current_state_index; // Index of the state in the action-state table -> Index of the subvector which contains the value of all action for that state
        long int future_state_index;
        std::vector<reward_type> current_action_values; // Vector with the value of all actions given the current state
        std::vector<reward_type> future_action_values; // Vector with the value of all actions given the current state
        std::vector<double> var_ranges;
        std::vector<int> disc_numbers; // Vector with the number of discretization points for each variable
        // Vector of length #action * #states. Matrix where rows are actions and columns states
        // The number of states can be quite large since the variables have to be discretized.
        // #states will therefore be the product of the discretization
        // point numbers of all variables
        std::vector<reward_type> qtable;
        long int state_space_size;
        size_t input_size;
        size_t num_mutations;

    public:
        /**
         * @brief constructor for agent_brain class
         * @post returns an object containing the ANN
         */


        agent_brain(unsigned int input_number_
                    ,unsigned int action_number_ // number of rows of the Q-table
                    , reward_type alpha_, reward_type gamma, double epsilon_ // learning parameters
                    , std::vector<int> disc_numbers // number of discretization value for each input
                    , std::vector<double> var_ranges_ // upper range of inputs, lower is 0
                    , std::string initial_attitude_="random");

       agent_brain(const agent_brain &parent_brain,const int &ID);
        void show_stats();     // print some information about the class to stdout
        void learn(const double &reward,const std::vector<double> &variables,const int &Timestep); // learn according to the reward (t time t-1) and the inputs (a time t)
        double get_reward(){return current_reward;}
        std::vector<reward_type> get_qtable(){return qtable;}
        // Get future state_values, Calculate index of new state and Set the possible future action values of the future state.
        int get_current_choice()const{return current_choice;}
        int get_future_choice()const{return future_choice;}
        std::vector<double> get_weights(const std::vector<double> inputs) const{
            return get_action_values(compute_state_index(compute_disc_indices(inputs)));}
        std::vector<double> test_input(const std::vector<double> input)const;
        void mutate(double noise, int m);
        void seed(std::vector<double> input,int action,double val);
        unsigned long int get_state_space_size(){return state_space_size;}
        unsigned long int get_num_mutations(){return num_mutations;}//Different brains will need different ammount of mutations
        void initialize_qtable(std::vector<reward_type>);
        void Q_learn();               // SARSA learning algorithm, update the Q-table
        void initialize_qtable(); // initialize Q-table
        unsigned int make_choice(std::vector<reward_type> values,const bool optimal=false) const;
        reward_type get_action_at(const int index,const int action) const {return qtable[index + state_space_size*action];}
        void set_action_at(int index, int action,double value) {qtable[index + state_space_size*action]=value;}
        std::vector<reward_type> get_action_values(const int index) const{
            std::vector<reward_type> retval;
            for (unsigned int i=0; i<action_number; i++) {
                retval.push_back(get_action_at(index,i));}
            return retval;}
        reward_type get_max_action_value(int index){
            reward_type retval(std::numeric_limits<reward_type>::lowest());
            reward_type current(0);
            for (unsigned int i=0; i<action_number; i++) {
                current=std::abs(get_action_at(index,i));
                if(current>retval){
                    retval=current;
                }
            }
            return retval;
        }
        void calc_state_space_size (){ // compute number of states (columns of q table) from number of inputs and discretization levels
            state_space_size = 1;
            for (unsigned int i=0; i<disc_numbers.size(); i++) {state_space_size *= disc_numbers[i];}}
        std::vector<int> compute_disc_indices(const std::vector<double> input)const;
        long int compute_state_index(const std::vector<int> disc_indices)const;

        // return a vector of vectors, each contains discretized values of the respective input
        // if the input is !=0, the provided value is the only value in the output
        std::vector<std::vector<int>> find_disc_values(std::vector<double> input) {
            int K = input.size();
            std::vector<std::vector<int>> retval(K);
            std::vector<int> disc_indices=compute_disc_indices(input);
            for (int i = 0; i < K; i++) {
                if(input[i]==0) {
                    for (int j = 0; j < disc_numbers[i]; j++) {
                        retval[i].push_back(j); // put all values
                    }
                } else {
                    retval[i].push_back(disc_indices[i]); // fix the value in the input
                }
            }
            return retval;
        }

        //TODO? No need to naively compute combinations
        // returns a vector of vector that contains all states defined by the input vector
        std::vector<std::vector<int>> compute_combinations(std::vector<double> input) {
            std::vector<std::vector<int>> vals = find_disc_values(input);
            int K = vals.size();
            std::vector<std::vector<int>> retval;
            std::vector<int> temp(K,0);
            // initialize each iterator to point to the beginning of each element of vals
            std::vector<std::vector<int>::iterator> iters(K); // iterators to vectors in vals
            for (unsigned int i =0; i < iters.size(); i++) {
                iters[i]=std::begin(vals[i]);
            }
            while (iters[0] != vals[0].end()) { // for all possible combinations of vals
                // find combination
                ++iters[K-1];
                for (int i = K-1; (i > 0) && (iters[i] == vals[i].end()); --i) {
                    iters[i] = vals[i].begin();
                    ++iters[i-1];
                }
                // for while loop to end the first iterator must be at the end of the vector, that creates random values
                if(iters[0] != vals[0].end()){
                    // generate the input vector
                    for (int i = 0; i < K; i++) {
                        temp[i]=*iters[i];
                    }
                    retval.push_back(temp);
                }
            }
            iters[0]=vals[0].begin(); // reset the first iterator and collect the last result
            // generate the input vector
            for (int i = 0; i < K; i++) {
                temp[i]=*iters[i];
            }
            retval.push_back(temp);
            return retval;
        }
    public:
        void print_Qtable(){
            std::vector<reward_type> acts;
            std::cout<<"|----------------|"<<std::endl;
            std::cout<<"|### Q TABLE ####|"<<std::endl;
            for(int i=0;i<state_space_size;i++){
                acts= get_action_values(i);
                std::cout<<"|# "<<i<<" ";
                print_table(acts);
            }
            std::cout<<"|----------------|"<<std::endl;
        }

        template<typename T1>
        void print_vector (std::vector<T1> vec) {
            std::string sep=" ";
            std::for_each(vec.begin(),vec.end(),[this,sep] (T1& v) { print_element(v,sep); });
            std::cout<<std::endl;}

        template<typename T1>
        void print_table (std::vector<T1> vec) {
            std::string sep="|";
            std::for_each(vec.begin(),vec.end(),[this,sep] (T1& v) { print_element(v,sep); });
            std::cout<<"|"<<std::endl;}

        template<typename T1>
        void print_element(T1 i,std::string sep) {std::cout << sep << i;}

        // This method lets cereal know which data members to serialize
        template<class Archive>
        void serialize(Archive & archive)
        {
          archive(CEREAL_NVP(qtable)); // serialize things by passing them to the archive
        }
    };
}      // namespace
#endif //! logic defined
