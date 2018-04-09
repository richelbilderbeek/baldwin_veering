/**
 * @file drl_logic.cpp
 * @brief implementation and declaration of the agent_brain
 */
#include <iostream>
#include <algorithm>
#include "rq_rl_logic.hpp"

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
        : rbm(input_number_,action_number_,1.5) //CHECK
        , action_number(action_number_)
        , alpha(alpha_)
        , gamma(gamma_)
        , epsilon(epsilon_)
        , initial_attitude(initial_attitude_)
        , dist(0,1) // Distribution for epsilon-greedy method
        , int_dist(0,action_number-1) // Distribution to choose action
        , prob01{std::bind(dist, std::ref(rng))}
        , randInt{std::bind(int_dist, std::ref(rng))}
        , past_state(disc_numbers_.size(),0)
        , past_action(0)
        , future_action(0)
        , past_state_index(-1)
        , future_state_index(0)
        , past_qvalues(action_number_)
        , future_qvalues(action_number_)
        , var_ranges(var_ranges_)
        , disc_numbers(disc_numbers_)
        , state_space_size(0)
        , input_size(input_number_)
        {
            print_table(var_ranges,false);
            std::cout<<" var ranges "<<std::endl;
            print_table(disc_numbers,false);
            std::cout<<" disc num "<<std::endl;
            calc_state_space_size();
            initialize_qtable();
            num_mutations=action_number*input_size;
        }

    agent_brain::agent_brain(const agent_brain &parent_brain,const int &ID)
        : rbm(parent_brain.rbm) //CHECK
        , action_number(parent_brain.action_number)
        , alpha(parent_brain.alpha)
        , gamma(parent_brain.gamma)
        , epsilon(parent_brain.epsilon)
        , initial_attitude(parent_brain.initial_attitude)
        , dist(parent_brain.dist) // Distribution for epsilon-greedy method
        , int_dist(parent_brain.int_dist) // Distribution to choose action
        , prob01{std::bind(dist, std::ref(rng))}
        , randInt{std::bind(int_dist, std::ref(rng))}
        , past_state(parent_brain.input_size,0)
        , past_action(0)
        , future_action(0)
        , past_state_index(-1)
        , future_state_index(0)
        , past_qvalues(parent_brain.action_number)
        , future_qvalues(parent_brain.action_number)
        , var_ranges(parent_brain.var_ranges)
        , disc_numbers(parent_brain.disc_numbers)
        , state_space_size(0)
        , input_size(parent_brain.input_size)
        , num_mutations(parent_brain.num_mutations)
        //Real constructor should copy memory and experiences
        {
            calc_state_space_size();

        }

    // This function should be called from the main program
    void agent_brain::learn(const double reward,const std::vector<double> input_state_,const int &Timestep){
        // Receive reward and observe new state;
        //
        //int TrainFrequency(1);
        std::vector<double> input_state(input_state_);
//        if(reward<=0)reward=-MAX_REWARD_DRL;
//        if(reward>0)reward=MAX_REWARD_DRL;
        //int TrainFrequency(1);

        for(auto &a:input_state){
            if(a>1)a=1;
        }

        future_action=make_choice(input_state,false);
        future_state_index = compute_state_index(compute_disc_indices(input_state));

        future_qvalues=rbm.get_weights(input_state);

        //Check convergence reusing old values and recalculating
        past_qvalues=rbm.get_weights(past_state);

#ifdef DEBUG
        std::cout<<" ACTION "<<past_action<<" ------ REWARD "<<reward<<std::endl;
        print_table(past_state,false);
        std::cout<<" PAST STATE"<<std::endl;
        print_table(input_state,false);
        std::cout<<" FUTURE STATE"<<std::endl;
        print_table(future_qvalues,false);
        std::cout<<" FUTURE QVAL- action "<<future_action<<std::endl;
        std::cout<<" Action "<<past_action <<" Reward "<<reward<<" gama "<<gamma<<" MaxQ "<<MaxQVal(future_qvalues)<<std::endl;
        print_table(past_qvalues,false);
        std::cout<<" BEFORE QVAL"<<std::endl;
#endif

        //Needs to be changed if terminal states are implemented
        past_qvalues[past_action]=reward+gamma*MaxQVal(future_qvalues);

#ifdef DEBUG
        print_table(past_qvalues,false);
        std::cout<<" AFTER "<<std::endl;
#endif
        int epochs(100);
#ifdef EPOCHS
        epochs=EPOCHS;
#endif
        rbm.train(past_state,past_qvalues,epochs);

#ifdef DEBUG
        std::vector<reward_type> temp_qvalues;
        temp_qvalues=rbm.get_QValues(past_state);
        print_table(temp_qvalues,false);
        std::cout<<" TRAINED "<<std::endl;
        std::cout<<" ------- \n"<<std::endl;
#endif
        // Move MDP to new state
        past_state_index = future_state_index;
        past_action = future_action;
        // Update action values
        past_qvalues=future_qvalues;
        // State
        past_state=input_state;
    }


    // Function to initialize action state matrix with values. Default is random
    void agent_brain::initialize_qtable () {
#ifdef DEBUG
        std::cout<<" state space "<<state_space_size<<" action number "<<action_number<<std::endl;
        std::cout<<" epsilon "<<epsilon<<" gamma "<<gamma<<" alpha "<<alpha<<std::endl;
#endif
        //Initializes the ANN reseting its weights and consequently q_values to 0
        rbm.Init();
    }


    // Overloaded function if customized values are given
    // NOT FUNCTIONAL YET
    void agent_brain::initialize_qtable(std::vector<reward_type> initial_values ) {
//        //Initializes the ANN reseting its weights and consequently q_values to 0
        rbm.Init();
        std::cout<<" UNIMPLEMENTED "<<std::endl;
        assert(1==0);
    }

    // Make choice with epsilon greedy method
    unsigned int agent_brain::make_choice(const std::vector<double> &input_state,const bool optimal) const {
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
        int qmax_id(0);
        double qmax(std::numeric_limits<reward_type>::lowest());
        std::vector<reward_type> temp_qvalues;
        std::vector<size_t> indexes(action_number,0);
        temp_qvalues=rbm.get_weights(input_state);
        for(size_t i=0;i<action_number;i++){
            indexes[i]= i;
        }
        std::random_shuffle(indexes.begin(),indexes.end());

        for(auto idx:indexes){
            if(temp_qvalues[idx]>qmax){
                qmax=temp_qvalues[idx];
                qmax_id=idx;
            }
        }
        return qmax_id;
       }
    }

    long int agent_brain::compute_state_index(std::vector<int> disc_indices) {
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

    std::vector<int> agent_brain::compute_disc_indices(std::vector<double> input) {
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

    /**
     * @brief show stats (alpha, epsilon, ...)
     */
    void agent_brain::show_stats(){
        std::cout << "state_space_size: " << state_space_size << std::endl;
        std::cout << "future_state_index: " << future_state_index << std::endl;
        std::cout << "Choice: " << future_action << std::endl;
        for (unsigned int i=0; i<action_number; i++) {
            std::cout << future_qvalues[i] << std::endl;
        }
    }

    // for a given input vector test the behavior of RL algorithm
    // value of variables related to elements !=0 in input vector is kept fixed.
    // for the elements=0 is taken the combination of all their possible discretized values.
    // the result is the avg of all rows in the Q-table that have value equal to the fixed elements.
    std::vector<double> agent_brain::test_input(std::vector<double> input)const{
        std::vector<double> input_state(input);
        for(auto &a:input_state){
            if(a>1)a=1;
        }
        std::vector<double> retval(action_number,0);
        retval=rbm.get_weights(input_state);
        return retval;
    }

    void agent_brain::mutate(double noise, int m) {
        assert((noise<=1)&&(noise>0));
       // Mutate an M random selection of states
        for(int i = 0; i < m; ++i ) {
            //double val=get_action_at(i1,i2);//NEEDS TO BE DISCUSSED
            double val_reward=MAX_REWARD;//NEEDS TO BE DISCUSSED
            //double val_reward=MAX_REWARD+gamma*MAX_REWARD;//NEEDS TO BE DISCUSSED
            double val_state=MAX_INPUT_VAL;//NEEDS TO BE DISCUSSED
            double range_reward=noise*val_reward;
            double range_state=noise*val_state;
            std::vector<double> q_val(action_number);
            std::vector<double> state(input_size);

            //This assumes 0 is the minimum input value
            std::uniform_real_distribution<double> distribution_for_q_values(-range_reward,range_reward);
            std::uniform_int_distribution<int> distribution_for_input_values(0,range_state);

            for(size_t i=0;i<input_size;i++){
                state[i]=std::max(std::min((double) distribution_for_input_values(rng),(double)MAX_INPUT_VAL),0.0);
            }
            for(size_t i=0;i<action_number;i++){
                //q_val[i]=std::max(std::min((double) distribution_for_q_values(rng),val_reward),-val_reward);
                q_val[i]=std::max(std::min((double) distribution_for_q_values(rng),(double)MAX_REWARD),(double) -MAX_REWARD);
            }
            rbm.train(state,q_val,1);
        }
    }

    // increase the cells corresponding to the inputs (combination of all elements=0, elements!=0 are fixed) and the action
    void agent_brain::seed(std::vector<double> input,int action,double val) {
        std::cout<<" UNIMPLEMENTED "<<std::endl;
        assert(1==0);
    }

    double agent_brain::MaxQVal(std::vector<double> q_values){
        //Breaks ties randomly
        double maxq=q_values[0];
        for(auto val:q_values){
            if(val>maxq) maxq=val;
        }
        return maxq;
    }
    bool agent_brain::add_update_experience(const std::vector<double> &state,const long int &state_idx,const int &action,const double &reward,const std::vector<double> &state_next,const long int state_next_idx,std::unordered_map<std::string,int> &seen_states,std::vector<Experience<std::vector<double>,int> > &memory_)const{
            std::stringstream ss1;
            ss1.clear();
            ss1.str("");
            ss1<<state_idx<<"_"<<action<<"_"<<state_next_idx;
            if(state_idx==-1)return false;
            if(seen_states.count(ss1.str())){
                //Updating old experience
                Experience<std::vector<double>,int> &exp_ref(memory_[seen_states[ss1.str()]]);
                exp_ref.reward=reward;
                return false;
            }else{
               //Adding new experience
               //std::cout<<"New experince "<<ss1.str()<<std::endl;
               seen_states[ss1.str()]=memory_.size();
               memory_.push_back(Experience<std::vector<double>,int>(state,state_idx,action,reward,state_next,state_next_idx));
               return true;
            }
            return false;
    }

    void agent_brain::action_replay(const std::vector<double> &past_state_,const long int &past_state_index_,const int &past_action_,const double &reward,const std::vector<double> &future_state_,const long int future_state_index_,const std::vector<Experience<std::vector<double>,int> > &memory){

        //Replay memory
        std::vector<Experience<std::vector<double>,int> > memory_replay(memory);
        uint max_memory_replay(10);
        std::random_shuffle(memory_replay.begin(),memory_replay.end());
        if(memory_replay.size()>max_memory_replay){
            memory_replay.resize(max_memory_replay);
        }
        std::vector<reward_type> q_values_CM,q_values_NM;
        std::vector< std::vector<double> > replay_states;
        std::vector< std::vector<reward_type> > replay_qvals;
        for(auto exper:memory_replay){
            q_values_CM=rbm.get_weights(exper.state);
            q_values_NM=rbm.get_weights(exper.state_next);
            q_values_CM[exper.action]=exper.reward+gamma*MaxQVal(q_values_NM);
            replay_states.push_back(exper.state);
            replay_qvals.push_back(q_values_CM);
            rbm.train(exper.state,q_values_CM,1);
        }
        //rbm.train_batch(replay_states,replay_qvals,1,1);
    }

    void agent_brain::temp_test(int i){
        std::vector<reward_type> temp_qvalues;
        std::vector<reward_type> input_state(input_size,1);
        temp_qvalues=rbm.get_weights(input_state);
        std::cout<<"\n TEST #"<<i<<std::endl;
        print_table(input_state,false);
        std::cout<<" INPUT"<<std::endl;
        print_table(temp_qvalues,false);
        std::cout<<" RESULT"<<std::endl;
    }
} // namespace
