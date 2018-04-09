/**
 * @file logic.cpp
 * @brief implementation and declaration of the agent_brain
 */
#include <iostream>
#include "pq_rl_logic.hpp"

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
        , last_state(input_number_,0)
        , future_state(input_number_,0)
        , current_action_values(action_number)
        , future_action_values(action_number)
        , var_ranges(var_ranges_)
        , disc_numbers(disc_numbers_)
        , state_space_size(0)
        , input_size(input_number_)
        {
            weights_.resize(action_number*input_size,0);

            //Currently bias unused (needs fix)
            bias_vis_.resize(input_size,1.0);
            bias_hid_.resize(action_number,1.0);
            num_mutations=action_number*input_size;
            //initialize_weights(); //Needs fix
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
        , last_state(parent_brain.last_state)
        , future_state(parent_brain.future_state)
        , current_action_values(parent_brain.current_action_values)
        , future_action_values(parent_brain.future_action_values)
        , var_ranges(parent_brain.var_ranges)
        , disc_numbers(parent_brain.disc_numbers)
        //, qtable(parent_brain.qtable)
        , weights_(parent_brain.weights_)
        , bias_vis_(parent_brain.bias_vis_)
        , bias_hid_(parent_brain.bias_hid_)
        , state_space_size(parent_brain.state_space_size)
        , input_size(parent_brain.input_size)
        , num_mutations(parent_brain.num_mutations)
        {
        }

     int agent_brain::choose_best_or_random(std::vector<double>  &qvalues){
            assert(qvalues.size()==action_number);
            std::vector<int> indexes(action_number,0);
            int qmax_id(0);
            double qmax(std::numeric_limits<double>::lowest());
            for(size_t i=0;i<action_number;i++){
                indexes[i]= i;
            }
            std::random_shuffle(indexes.begin(),indexes.end());
            for(auto idx:indexes){
                if(qvalues[idx]>qmax){
                    qmax=qvalues[idx];
                    qmax_id=idx;
                }
            }
            return qmax_id;
     }

     std::vector<double> agent_brain::back_activate(std::vector<double> outputs){
        assert(outputs.size()==action_number);
        std::vector<double> retval(input_size,0);
         for (size_t i = 0; i < input_size; i++)
            for (size_t j=0; j < action_number; j++)
                retval[i]+=weights_[i*action_number+j]*outputs[j];
        return retval;
    }
    void agent_brain::tensor_prod(std::vector<double> &inputs,std::vector<double> &outputs, std::vector<double> &grad){
        assert(inputs.size()==input_size);
        assert(outputs.size()==action_number);
//        for (size_t j=0; j < action_number; j++)
//            for (size_t i = 0; i < input_size; i++)
//                grad[i*(action_number)+j]=inputs[i]*outputs[j];
        //Needs discussion
        for (size_t j=0; j < action_number; j++)
            for (size_t i = 0; i < input_size; i++){
                if(fabs(inputs[i])>std::numeric_limits<double>::epsilon()){
                    grad[i*(action_number)+j]=outputs[j]/inputs[i];
                }else{
                    grad[i*(action_number)+j]=0;
                }
            }
    }

    void agent_brain::apply_cd(std::vector<double> &weight, std::vector<double> &pos_grad, std::vector<double> &neg_grad,double lrate){
        assert(pos_grad.size()==(action_number*(input_size)));
        assert(neg_grad.size()==(action_number*(input_size)));
        for (size_t i = 0; i < input_size; i++)
            for (size_t j=0; j < action_number; j++)
                weight[i*(action_number)+j]-=lrate*(pos_grad[i*(action_number)+j]-neg_grad[i*(action_number)+j]);
    }

    void agent_brain::learn(const double &reward_  /*!< The reward of the last timestep */
                            ,const std::vector<double> &input_state,  /*!< A perception vector */
                            const int &Timestep                       // not used but kept for compatibility with other implementations
                            ){
        double last_reward(reward_);
        std::vector<double> input(input_state);

        for(auto &a:input){
            if(a>1)a=1;
        }
        future_state=input;

        future_choice=make_choice(input,false);

        std::vector<double> future_qvalues=get_weights(future_state);
        std::vector<double> current_qvalues=get_weights(last_state);

        std::vector<double> pos_grad(action_number*(input_size),0);
        std::vector<double> neg_grad(action_number*(input_size),0);

        tensor_prod(last_state,current_qvalues,pos_grad);

        int best_future(choose_best_or_random(future_qvalues));
        current_qvalues[current_choice]=last_reward+gamma*future_qvalues[best_future];
        tensor_prod(last_state,current_qvalues,neg_grad);
        apply_cd(weights_,pos_grad,neg_grad,0.1);
        current_choice=future_choice;
        last_state=future_state;
    }
    // Function to initialize action state matrix with values. Default is random
    void agent_brain::initialize_weights (){
        //! Initialize the genome randomly with values between 0 and 1
        std::uniform_real_distribution<double> distribution_for_weight_values(0.,1.);
        for (size_t i = 0; i < input_size; i++)
            for (size_t j=0; j < action_number; j++)
                weights_[i*(action_number)+j] = distribution_for_weight_values(rng);
    }

    //Make choice with epsilon greedy method
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
        temp_qvalues=get_weights(input_state);
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

    std::vector<double> agent_brain::get_weights(const std::vector<double> inputs)const{
        assert(inputs.size()==input_size);
        std::vector<double> retval(action_number,0);
        for (size_t i = 0; i < input_size; i++)
            for (size_t j=0; j < action_number; j++)
                retval[j]+=weights_[i*(action_number)+j]*inputs[i];
        return retval;
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
        retval=get_weights(input_state);
        return retval;
    }

    void agent_brain::mutate(double noise, int m) {
        assert((noise<=1)&&(noise>0));
        double val(0);
        std::uniform_int_distribution<int> distribution_for_weights_1(0,input_size-1);
        std::uniform_int_distribution<int> distribution_for_weights_2(0,action_number-1);
        double max_val(MAX_WEIGHT);
        double range = noise;
        std::uniform_real_distribution<double> distribution_for_weight_ranges(-range,range);

        //std::cout <<" noise"<<noise<<" m "<<m<<std::endl;

        // Mutate a random selection of M weights
        for( int i = 0; i < m; ++i ) {
            int i1 = distribution_for_weights_1(rng);
            int i2 = distribution_for_weights_2(rng);


            if(std::abs(weights_[i1*action_number+i2])>0.0){
                val = std::max(std::min(weights_[i1*action_number+i2]+distribution_for_weight_ranges(rng)*weights_[i1*action_number+i2],(double)MAX_WEIGHT),(double)-MAX_WEIGHT);
            }else{
                val = std::max(std::min(weights_[i1*action_number+i2]+distribution_for_weight_ranges(rng)*max_val,(double)MAX_WEIGHT),(double)-MAX_WEIGHT);
            }
            //std::cout <<" val "<<val<<" prev "<<weights_[i1*action_number+i2]<<std::endl;
            weights_[i1*action_number+i2] = val;
        }
    }

    void agent_brain::seed(std::vector<double> input,int action,double val) {
        assert(input.size()==input_size);
        for(size_t i=0; i<input_size;i++)
            if(input[i]!=0)
                weights_[i*(action_number)+action]=val;
    }

    template<typename T1>
    void print_element(T1 i,std::string sep) {std::cout << sep << i;}

    template<typename T1>
    void agent_brain::print_table(std::vector<T1> vec){
            std::string sep="|";
            std::for_each(vec.begin(),vec.end(),[this,sep] (T1& v) { print_element(v,sep); });
            std::cout<<"|"<<std::endl;}

    void agent_brain::print_weights(std::vector<double> &table){
        assert(table.size()==(input_size*action_number));
        for (size_t i = 0; i < input_size; i++){
            std::cout<<"|";
            for (size_t j=0; j < action_number; j++){
                std::cout<<table[i*(action_number)+j]<<"|";
            }
            std::cout<<std::endl;
        }
    }
} // namespace
