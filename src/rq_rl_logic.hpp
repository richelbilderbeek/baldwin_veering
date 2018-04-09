/**
 * @file drl_logic.hpp
 * @brief containing the deep reinforcement learning logic for the agents "brain"
 * based on rl_logic.hpp
 * @author: Leonel Aguilar - leonel.aguilar.m/at/gmail.com
 */

#ifndef RL_LOGIC
#define RL_LOGIC
#include <iostream>

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
#include "rbm.hpp"
#include "constants.h"
#include "cereal/cereal.hpp"
//#define DEBUG

namespace Joleste
{

#ifndef RNG
#define RNG
    extern std::mt19937 rng;
#endif
    template<typename T1,typename T2>
    class Experience{
    public:
        T1 state;
        long int state_idx;
        T2 action;
        double reward;
        T1 state_next;
        long int state_next_idx;

        Experience(T1 state_,long int state_idx_, T2 action_,double reward_,T1 state_next_,long int state_next_idx_):
        state(state_), state_idx(state_idx_), action(action_),reward(reward_),state_next(state_next_),state_next_idx(state_next_idx_)
        {}
        Experience():
        state_idx(-1), action(-1),reward(-1),state_next(-1),state_next_idx(-1)
        {}
    };

    class agent_brain
    {

    public:
        /**
         * @brief constructor for agent_brain class
         * @post returns an object containing the ANN
         */
        typedef double reward_type;

        agent_brain( unsigned int input_number_
                    ,unsigned int action_number_
                    , reward_type alpha_, reward_type gamma, double epsilon_ // learning parameters
                    , std::vector<int> disc_numbers // number of discretization value for each input
                    , std::vector<double> var_ranges_ // upper range of inputs, lower is 0
                    , std::string initial_attitude_="random");

        agent_brain(const agent_brain &parent_brain,const int &ID);

        //~agent_brain(void); /// provided by compiler

        void show_stats();     // print some information about the class to stdout
        void learn(const double reward,const std::vector<double> input_state,const int &Timestep); // learn according to the reward (t time t-1) and the inputs (a time t)
        // Get future state_values, Calculate index of new state and Set the possible future action values of the future state.
        int get_current_choice()const{return past_action;}
        int get_future_choice()const{return future_action;}
        std::vector<double> get_weights(std::vector<double> inputs)const{
            return rbm.get_weights(inputs);}
        std::vector<double> test_input(std::vector<double> input)const;
        void mutate(double noise, int m);
        void seed(std::vector<double> input,int action,double val);
        unsigned long int get_state_space_size(){return state_space_size;}
        unsigned long int get_num_mutations(){return num_mutations;}//Different brains will need different ammount of mutations
        double MaxQVal(std::vector<double> q_values);

        //For action replay
        void action_replay(const std::vector<double> &state,const long int &state_idx,const int &action,const double &reward,const std::vector<double> &state_next,const long int state_next_idx,const std::vector<Experience<std::vector<double>,int> > &memory_);
        bool add_update_experience(const std::vector<double> &state,const long int &state_idx,const int &action,const double &reward,const std::vector<double> &state_next,const long int state_next_idx,std::unordered_map<std::string,int> &seen_states,std::vector<Experience<std::vector<double>,int> > &memory_)const;

    private:
        // parameters
    public://for Debugging
        RBM_DBN::RBM rbm;
        unsigned int action_number;
        reward_type alpha;
        reward_type gamma;
        double epsilon;
        std::string initial_attitude;
        std::uniform_real_distribution<double> dist;
        std::uniform_int_distribution<int> int_dist;
        std::function <double()> prob01;
        std::function <int()> randInt;
        std::vector<double> past_state;
        int past_action;
        int future_action;
        long int past_state_index; // Index of the state in the action-state table -> Index of the subvector which contains the value of all action for that state
        long int future_state_index;
        std::vector<reward_type> past_qvalues; // Vector with the value of all actions given the current state
        std::vector<reward_type> future_qvalues; // Vector with the value of all actions given the current state
        std::vector<double> var_ranges;
        std::vector<int> disc_numbers; // Vector with the number of discretization points for each variable
        long int state_space_size;
        size_t input_size;
        size_t num_mutations;
        std::vector<Experience<std::vector<double>,int> > memory_;
        std::unordered_map<std::string,int> seen_states_;

        void initialize_qtable(std::vector<reward_type>);
        void initialize_qtable(); // initialize Q-table
        unsigned int make_choice(const std::vector<double> &input_state,const bool optimal=false) const;
        void calc_state_space_size (){ // compute number of states (columns of q table) from number of inputs and discretization levels
            state_space_size = 1;
            for (unsigned int i=0; i<disc_numbers.size(); i++) {state_space_size *= disc_numbers[i];}}
        std::vector<int> compute_disc_indices(std::vector<double> input);
        long int compute_state_index(std::vector<int> disc_indices);

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


        template<typename T1>
        void print_vector (std::vector<T1> vec) {
            std::string sep=" ";
            std::for_each(vec.begin(),vec.end(),[this,sep] (T1& v) { print_element(v,sep); });
            std::cout<<std::endl;}

        template<typename T1>
        void print_table (std::vector<T1> vec,bool lineEnd=true) {
            std::string sep="|";
            std::for_each(vec.begin(),vec.end(),[this,sep] (T1& v) { print_element(v,sep); });
            std::cout<<"|";
            if(lineEnd)
                std::cout<<std::endl;
        }
        template<typename T1>
        void print_element(T1 i,std::string sep) {std::cout << sep << i;}
    public:
        void temp_test(int i);

        // This method lets cereal know which data members to serialize
        template<class Archive>
        void serialize(Archive & archive)
        {
          archive(CEREAL_NVP(rbm)); // serialize things by passing them to the archive
        }
    };


}      // namespace
#endif //! logic defined
