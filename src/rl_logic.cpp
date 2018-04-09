/**
 * @file logic.cpp
 * @brief implementation and declaration of the agent_brain
 */
#include <iostream>
#include "rl_logic.hpp"

template <typename T>
std::vector<T> operator*(const std::vector<T>& a, const T& b)
{

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), std::back_inserter(result),
                   std::bind2nd(std::multiplies<T>(),b));
    return result;
}

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}
template <typename T>
std::vector<T> operator/(const std::vector<T>& a, const T& b)
{

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), std::back_inserter(result),
                   std::bind2nd(std::divides<T>(),b));
    return result;
}

namespace Joleste
{
    agent_brain::agent_brain(unsigned int input_number_,unsigned int action_number_, reward_type alpha_, reward_type gamma_, double epsilon_, std::vector<int> disc_numbers_, std::vector<double> var_ranges_, std::string initial_attitude_)
        : action_number(action_number_)
        , alpha(alpha_)
        , gamma(gamma_)
        , epsilon(epsilon_)
        , initial_attitude(initial_attitude_)
        , dist(0,1) // Distribution for epsilon-greedy method
        , int_dist(0,action_number-1) // Distribution to choose action
        , prob01{std::bind(dist, std::ref(rng))}
        , randInt{std::bind(int_dist, std::ref(rng))}
        , current_choice(0)
        , future_choice(0)
        , current_reward(0)
        , current_state_index(0)
        , future_state_index(0)
        , current_action_values(action_number)
        , future_action_values(action_number)
        , var_ranges(var_ranges_)
        , disc_numbers(disc_numbers_)
        , state_space_size(0)
        , input_size(input_number_)
        {
            //std::swap(disc_numbers, disc_numbers_);
            calc_state_space_size();
            initialize_qtable();
            num_mutations=state_space_size; //Number of mutations
            // initialize action state counters
            //qtable_counters.resize(state_space_size,0);
        }
    //Replicate constructor
    agent_brain::agent_brain(const agent_brain &parent_brain,const int &ID)
        : action_number(parent_brain.action_number)
        , alpha(parent_brain.alpha)
        , gamma(parent_brain.gamma)
        , epsilon(parent_brain.epsilon)
        , initial_attitude(parent_brain.initial_attitude)
        , dist(parent_brain.dist) // Distribution for epsilon-greedy method
        , int_dist(0,parent_brain.action_number-1) // Distribution to choose action
        , prob01{std::bind(dist, std::ref(rng))}
        , randInt{std::bind(int_dist, std::ref(rng))}
        , current_choice(parent_brain.current_choice)
        , future_choice(parent_brain.future_choice)
        , current_reward(parent_brain.current_reward)
        , current_state_index(parent_brain.current_state_index)
        , future_state_index(parent_brain.future_state_index)
        , current_action_values(parent_brain.current_action_values)
        , future_action_values(parent_brain.future_action_values)
        , var_ranges(parent_brain.var_ranges)
        , disc_numbers(parent_brain.disc_numbers)
        , qtable(parent_brain.qtable)
        , state_space_size(parent_brain.state_space_size)
        , input_size(parent_brain.input_size)
        , num_mutations(parent_brain.num_mutations)
        {
        }


    // This function should be called from the main program
    void agent_brain::learn (const double &reward_,const std::vector<double> &input_state,const int &Timestep) {
        double reward(reward_);
        std::vector<double> input_state_(input_state);

        for(auto &a:input_state_){
            if(a>1)a=1;
        }

        current_reward = reward;
        future_state_index = compute_state_index(compute_disc_indices(input_state_)); // Calculates disc_indices by discretizing the state values with disc_numbers
        future_action_values = get_action_values(future_state_index); // Get a vector with the values of every possible action in the future state

        future_choice=make_choice(input_state_);
#ifdef DEBUG
        std::cout<<"Current table entry ("<<future_state_index<<"): ";
        std::vector<reward_type> acts = get_action_values(future_state_index);
        print_table(acts);
        std::cout<<"choice is: "<<future_choice<<", old choice was: "<<current_choice<<std::endl;
        std::cout<<"Initial table entry ("<<current_state_index<<"): ";
        acts = get_action_values(current_state_index);
        print_table(acts);
#endif
        Q_learn(); // Update Q according to SARSA formula
#ifdef DEBUG
        std::cout<<"Final table entry ("<<current_state_index<<"): |";
        acts = get_action_values(current_state_index);
        print_table(acts);
#endif
        // update state counter
        //qtable_counters[current_state_index]++;
        // Move MDP to new state
        current_state_index = future_state_index;
        current_choice = future_choice;
        // Update action values
        std::swap(current_action_values, future_action_values);
    }


    // Function to initialize action state matrix with values. Default is random
    void agent_brain::initialize_qtable () {
        assert(qtable.size()==0);
#ifdef DEBUG
        std::cout<<" state space "<<state_space_size<<" action number "<<action_number<<std::endl;
        std::cout<<" epsilon "<<epsilon<<" gamma "<<gamma<<" alpha "<<alpha<<std::endl;
#endif
        qtable.reserve(state_space_size*action_number); // prevents segfaults due to bad allocation
        // Optimistic choice encourages exploration in the beginning
        if (initial_attitude.compare("optimistic") == 0) {
            qtable.resize(state_space_size*action_number,1.0);
        }
        // Pessimistic choice encourages exploitation in the beginning
        else if  (initial_attitude.compare("pessimistic") == 0) {
            qtable.resize(state_space_size*action_number,0.0);
        }
        else { // Random initial values
            for (unsigned int i=0; i<state_space_size*action_number; i++) {
                qtable.push_back(prob01()); // generate a new random value at every iteration
            }
        }
    }


    // Overloaded function if customized values are given
    void agent_brain::initialize_qtable (std::vector<reward_type> initial_values ) {
        if (initial_attitude.compare("customized") == 0) {
            std::swap(qtable,initial_values);
        }
        else {
            std::cout << "Warning! No customized option given!" << std::endl;
            std::swap(qtable,initial_values);
        }
    }
    // Make choice with epsilon greedy method
    unsigned int agent_brain::make_choice(const std::vector<double> input_state,const bool optimal) const{
        if ((!optimal)&&(prob01() < epsilon)) { // Pick random action -> explore
#ifdef DEBUG
        std::cout << "##EXPLORE" << std::endl;
#endif
            return randInt();
        }
        else {                  // Get action with maximum payoff -> exploit
#ifdef DEBUG
        std::cout << "$$EXPLOIT" << std::endl;
#endif
            std::vector<size_t> indexes(action_number,0);
            int qmax_id(0);
            double qmax(std::numeric_limits<reward_type>::lowest());
            for(size_t i=0;i<action_number;i++){
                indexes[i]= i;
            }
            std::random_shuffle(indexes.begin(),indexes.end());

            long int state_index = compute_state_index(compute_disc_indices(input_state)); // Calculates disc_indices by discretizing the state values with disc_numbers
            std::vector<reward_type> values = get_action_values(state_index); // Get a vector with the values of every possible action in the future state

            for(auto idx:indexes){
                if(values[idx]>qmax){
                    qmax=values[idx];
                    qmax_id=idx;

                }
            }
            return qmax_id;
        }
    }

    long int agent_brain::compute_state_index(const std::vector<int> disc_indices)const{
        long int retval = disc_indices[0];
        for (unsigned int i=1; i<disc_numbers.size(); i++) {
            int temp = disc_indices[i];
            for(unsigned int j=0; j<i; j++) {
                temp *= disc_numbers[j];
            }
            retval += temp;
        }
        return retval;
    }

    std::vector<int> agent_brain::compute_disc_indices(const std::vector<double> input)const{
        std::vector<int> retval(disc_numbers.size()); // Vector with the indices of the current discretized variables in the discretized range
        for (unsigned int i=0; i<disc_numbers.size(); i++){ // Over all states. #states = disc_ number.size()
            if(input[i]>var_ranges[i]){
                retval[i] = (int(disc_numbers[i]-0.00000001));
                //std::cout<<"#i "<<i<<" input[i] "<<input[i]<<" var_ranges[i] "<<var_ranges[i]<<" disc_numbers[i] "<<disc_numbers[i]<<std::endl;
            }else{
                retval[i] = (int(input[i]*disc_numbers[i]/var_ranges[i]-0.00000001));
                //std::cout<<"#i "<<i<<" input[i] "<<input[i]<<" var_ranges[i] "<<var_ranges[i]<<" disc_numbers[i] "<<disc_numbers[i]<<std::endl;
            }
        }
        return retval;
    }

    // Q function for epsilon greedy learning. No estimation of future reward!
    void agent_brain::Q_learn () {
        int current_index = current_state_index + state_space_size*current_choice;
#ifdef SARSA
        int future_index = future_state_index + state_space_size*future_choice;
#else //QLearning
        //Best future choice for Qlearning
  int best_future_choice = std::minmax_element(future_action_values.begin(), future_action_values.end()).second-future_action_values.begin();
        int future_index = future_state_index + state_space_size*best_future_choice;//Qlearning
#endif
        qtable[current_index] += alpha*(current_reward + gamma*qtable[future_index] - qtable[current_index]);
    }

    /**
     * @brief show stats (alpha, epsilon, ...)
     */
    void agent_brain::show_stats(){
        std::cout << "state_space_size: " << state_space_size << std::endl;
        std::cout << "future_state_index: " << future_state_index << std::endl;
        std::cout << "reward: " << current_reward << std::endl;
        std::cout << "Choice: " << future_choice << std::endl;
        for (unsigned int i=0; i<action_number; i++) {
            std::cout << future_action_values[i] << std::endl;
        }
        for (unsigned int i=0; i<action_number; i++) {
            std::cout << get_action_at(future_state_index,i) << std::endl;
        }
    }

    // for a given input vector test the behavior of RL algorithm
    // value of variables related to elements !=0 in input vector is kept fixed.
    // for the elements=0 is taken the combination of all their possible discretized values.
    // the result is the avg of all rows in the Q-table that have value equal to the fixed elements.
    std::vector<double> agent_brain::test_input(const std::vector<double> input)const{
        std::vector<double> input_state(input);
        for(auto &a:input_state){
            if(a>1)a=1;
        }
        std::vector<double> retval(action_number,0); // contains the average response
        retval=get_action_values(compute_state_index(compute_disc_indices(input_state)));
        return retval;
    }

    void agent_brain::mutate(double noise, int m) {
        assert((noise<=1)&&(noise>0));
        std::uniform_int_distribution<int> distribution_for_weights_1(0,state_space_size-1);
        std::uniform_int_distribution<int> distribution_for_weights_2(0,action_number-1);

        double val(0);
        double max_val(0.001*MAX_WEIGHT);
        double range = noise;
        std::uniform_real_distribution<double> distribution_for_weight_ranges(-range,range);
        //std::cout <<" noise"<<noise<<" m "<<m<<std::endl;
        // Mutate a random selection of M weights
        for(int i = 0; i < m; ++i ) {
            int i1 = distribution_for_weights_1(rng);
            int i2 = distribution_for_weights_2(rng);
            if(std::abs(get_action_at(i1,i2))>0.0){
                val = std::max(std::min(get_action_at(i1,i2)+distribution_for_weight_ranges(rng)*get_action_at(i1,i2),(double)MAX_WEIGHT),(double)-MAX_WEIGHT);
            }else{
                val = std::max(std::min(get_action_at(i1,i2)+distribution_for_weight_ranges(rng)*max_val,(double)MAX_WEIGHT),(double)-MAX_WEIGHT);
            }
            //std::cout <<" val "<<val<<" prev "<<weights_[i1*action_number+i2]<<std::endl;
            set_action_at(i1,i2,val);
        }
    }

    // increase the cells corresponding to the inputs (combination of all elements=0, elements!=0 are fixed) and the action
    void agent_brain::seed(std::vector<double> input,int action,double val) {
        // find indices of all combinations
        std::vector<std::vector<int>> combs = compute_combinations(input);
        // update weights
        std::for_each(combs.begin(),combs.end(),[action,this,val](std::vector<int> i){
                set_action_at(compute_state_index(i),action,val);});
    }
} // namespace
