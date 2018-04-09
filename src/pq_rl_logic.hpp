/**
 * @file rl_logic.hpp
 * @brief containing the reinforcement learning logic for the agents "brain"
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
#include "constants.h"
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


        typedef double reward_type;

    private:
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
        std::vector<double> last_state;
        std::vector<double> future_state;
        std::vector<reward_type> current_action_values; // Vector with the value of all actions given the current state
        std::vector<reward_type> future_action_values; // Vector with the value of all actions given the current state
        std::vector<double> var_ranges;
        std::vector<int> disc_numbers; // Vector with the number of discretization points for each variable
        // Vector of length #action * #states. Matrix where rows are actions and columns states
        // The number of states can be quite large since the variables have to be discretized.
        // #states will therefore be the product of the discretization
        // point numbers of all variables
        std::vector<double> weights_; /*!< The genome (brain weights) of the agent */
        std::vector<double> bias_vis_;
        std::vector<double> bias_hid_;
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
        //replicate constructor
        agent_brain(const agent_brain &parent_brain,const int &ID);
        void show_stats();     // print some information about the class to stdout
        int choose_best_or_random(std::vector<double>  &qvalues);
        std::vector<double> back_activate(std::vector<double> outputs);
        void tensor_prod(std::vector<double> &inputs,std::vector<double> &outputs, std::vector<double> &grad);
        void apply_cd(std::vector<double> &weight, std::vector<double> &pos_grad, std::vector<double> &neg_grad,double lrate);
        void learn(const double &last_reward,const std::vector<double> &input,const int &Timestep);
        double get_reward(){return current_reward;}
        // Get future state_values, Calculate index of new state and Set the possible future action values of the future state.
        int get_current_choice() const{return current_choice;}
        int get_future_choice() const {return future_choice;}
        std::vector<double> get_weights(const std::vector<double> inputs)const;
        std::vector<double> test_input(const std::vector<double> input)const;
        void mutate(double noise, int m);
        void seed(std::vector<double> input,int action,double val);
        unsigned long int get_state_space_size(){return state_space_size;}
        unsigned long int get_num_mutations(){return num_mutations;}//Different brains will need different ammount of mutations
        unsigned int make_choice(const std::vector<double> &input_state,const bool optimal) const;

        void print_weights(std::vector<double> &table);
        template<typename T1>
        void print_table(std::vector<T1> vec);
        void initialize_weights(); // initialize RBM weights

    public:
        // This method lets cereal know which data members to serialize
        template<class Archive>
        void serialize(Archive & archive)
        {
          archive(CEREAL_NVP(weights_)); // serialize things by passing them to the archive
        }
    };
}      // namespace
#endif //! logic defined
