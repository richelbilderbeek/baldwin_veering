#include "genome.hpp"
#include <sstream>


namespace Joleste
{
    std::mt19937 rng(rand()); /// rng for the whole simulation. Available in the whole namespace!

    double Genome::weight_mutation_rate_ = 0.5;

    Genome::Genome(double (&skill_level)[FOOD_TYPES],float mutation_size_skill_,float mutation_size_weigths_)
        : var_ranges_(N_PERCEPTIONS,1)
        , disc_numbers_(N_PERCEPTIONS,2)
        , mutation_size_skill(mutation_size_skill_)
        , mutation_size_weights(mutation_size_weigths_)
        , brain_(N_PERCEPTIONS,N_OUTPUTS,rl_alpha,rl_gamma,rl_epsilon,disc_numbers_,var_ranges_,"pessimistic")

    {
        assert((mutation_size_skill<=1)&&(mutation_size_skill>0));
        assert((mutation_size_weights<=1)&&(mutation_size_weights>0));
        for (int i=0;i<FOOD_TYPES;i++){
            skill_level_[i]=skill_level[i];
        }

#ifdef DEBUG
        std::cout<<"GENOME DISC SIZE "<<disc_numbers_.size()<<" VAR RANGE "<<var_ranges_.size()<<std::endl;
        std::cout<<"DISC SIZES ";
        for(size_t i=0;i<disc_numbers_.size();i++){
            std::cout<<disc_numbers_[i]<<" ";
        }
        std::cout<<std::endl;

        std::cout<<"VAR RANGES ";
        for(size_t i=0;i<var_ranges_.size();i++){
            std::cout<<var_ranges_[i]<<" ";
        }
        std::cout<<std::endl;
#endif
    }

    Genome::Genome()
        : var_ranges_(N_PERCEPTIONS,1)
        , disc_numbers_(N_PERCEPTIONS,2)
        , mutation_size_skill(1)
        , mutation_size_weights(1)
        , brain_(N_PERCEPTIONS,N_OUTPUTS,rl_alpha,rl_gamma,rl_epsilon,disc_numbers_,var_ranges_,"pessimistic")
    {
        assert((mutation_size_skill<=1)&&(mutation_size_skill>0));
        assert((mutation_size_weights<=1)&&(mutation_size_weights>0));
        //Mutate skill levels
        std::uniform_real_distribution<double> distribution_for_weight_values(0.,1.);
        assert(FOOD_TYPES==2);
        skill_level_[0]=distribution_for_weight_values(rng);
        skill_level_[1]=1-skill_level_[0];
        assert(skill_level_[0]+skill_level_[1]==1);
    }

    //Replicate constructor
    Genome::Genome(const Genome &parent_gen,int ID)
        : var_ranges_(parent_gen.var_ranges_)
        , disc_numbers_(parent_gen.disc_numbers_)
        , mutation_size_skill(parent_gen.mutation_size_skill)
        , mutation_size_weights(parent_gen.mutation_size_weights)
        , brain_(parent_gen.brain_,ID)
        //Check all
    {
        assert((mutation_size_skill<=1)&&(mutation_size_skill>0));
        assert((mutation_size_weights<=1)&&(mutation_size_weights>0));
        for (int i=0;i<FOOD_TYPES;i++){
            skill_level_[i]=parent_gen.skill_level_[i];
        }
    }

    //Genome
    void Genome::mutate(size_t m){
        std::uniform_real_distribution<double> distribution_for_weight_values(-1.,1.);
        //Mutate skill levels
        assert(FOOD_TYPES==2);
        increase_skill(0,mutation_size_skill*distribution_for_weight_values(rng));
#ifndef NMUTATE
        brain_.mutate(mutation_size_weights,m);
#endif
    }

    void Genome::mutate()
    {
        mutate(weight_mutation_rate_*brain_.get_num_mutations());
    }

    // compute and return the score of possible actions for the current state
    Genome::perception_type Genome::activate(const Genome::perception_type inputs)const {
        assert(inputs.size()==N_PERCEPTIONS);
        return brain_.get_weights(inputs);}

    // compute and return the score of possible actions for the current state
    // update Q-table with reward from last round
    void Genome::train(const perception_type &inputs,const double &last_reward,const int &Timestep) {
        assert(inputs.size()==N_PERCEPTIONS);
#ifdef DEBUG
        std::stringstream inps;
        inps<<("|");
        for (auto i = inputs.begin(); i != inputs.end(); i++) {
            inps<<*i<<"|";
        }
        std::cout <<"getting trained on input " << inps.str() <<" and feedback "<<last_reward<<std::endl;
#endif
        // give to SARSA/QLearning the reward given by the previous action and the current state
        // Sarsa will compute the best action given the current input and update the table for the previous action with the previous feedback
        brain_.learn(last_reward,inputs,Timestep);
    }

    int Genome::decision_algorithm(const perception_type &perception)const{

#ifdef LEARN
        //Best but couples Learning and evolution
        return brain_.get_future_choice();
#else
        // perception_type retval=activate(perception);
        // int action =choose_best_or_random(retval);
        int action =choose_best_or_random(perception);
        return action;
#endif
    }

    Genome::actions_type Genome::test_input(const Genome::perception_type input)const{
        std::vector<double> retval(brain_.test_input(input));
        return retval;
    }

    void Genome::seed(Genome::perception_type input,int action,double val) {
        brain_.seed(input,action,val);}

} // end namespace Joleste
