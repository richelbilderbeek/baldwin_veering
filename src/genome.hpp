#ifndef GENOME_HPP
#define GENOME_HPP

#include "constants.h"

#if defined(BRAIN_QL)
#include "rl_logic.hpp"
#elif defined(BRAIN_DEEP)
#include "dq_rl_logic.hpp"
#elif defined(BRAIN_RQL)
#include "rq_rl_logic.hpp"
#elif defined(BRAIN_PQL)
#include "pq_rl_logic.hpp"
#else
#warning no brain defined
#endif

#include <vector> // we could use a list instead of vector, since for high orders the list is quicker and more efficient. If we just need up to 100 i think vector is more easy to use. If we need to switch the weights, then list is also preferable
#include <limits>
#include <random>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <valarray>     // std::valarray
#include "cereal/cereal.hpp"


namespace Joleste
{
    //! Define the random number generator for the whole simulation
#ifndef RNG
#define RNG
    extern std::mt19937 rng;
#endif

    typedef unsigned age_type; // define age_type globally in the whole namespace

    //!  Genome class
    /*!
      This class contains the brain (genetic description) of an agent and the functions for replication and mutation
    */
    class Genome
    {
    public:

        typedef std::vector<double> perception_type; /*!< defines the format of perceptions */
        typedef std::vector<double> actions_type; /*!< defines the format of actions */

    private:

        std::vector<double> var_ranges_; // TODO is this used here, or only in RL?
        std::vector<int> disc_numbers_;// TODO is this used here, or only in RL?
        void mutate(size_t m);
        static double weight_mutation_rate_;
        double TEMPERATURE; //! TODO unused
        float mutation_size_skill;
        float mutation_size_weights;
        double skill_level_[FOOD_TYPES];
#ifdef TEST
    public:     // test suite needs the brain to be public
#endif
        agent_brain brain_;



    public:
        static void set_weight_mutation_rate( double m){weight_mutation_rate_ = m;}

        Genome(double (&skill_level)[FOOD_TYPES],float mutation_size_skill_,float mutation_size_weigths_);
        Genome();
        Genome(const Genome &gen_parent,int ID);
        void mutate();
        void mutate_high();
        void set_temp(double x){TEMPERATURE=x;} // temperature is used in the mutation with decay
        void mutate_decay(size_t t);
        perception_type activate(const perception_type inputs)const; /*!< Produce a decision from the inputs */
        void train(const perception_type &inputs,const double &last_reward,const int &Timestep); /*!< dummy function, will be used for learning */
        int best_action(const perception_type &perception) const{
            return brain_.make_choice(perception,true);
        }
        int decision_algorithm(const perception_type &perception)const;
        int choose_best_or_random(const perception_type &perception)const{
            return brain_.make_choice(perception,false);
        }
        std::string prettyprint_weights(perception_type inputs); /*!< utility to print the genome */
//        double get_weights(size_t i, size_t j) {assert(i<N_PERCEPTIONS); assert(j<N_OUTPUTS); return weights_[i][j];};
        double return_temp(){ return TEMPERATURE;} /*!< TODO unused */               // TODO rename to get_temp()
        double return_skill(const size_t idx)const{assert(idx<FOOD_TYPES);return skill_level_[idx];};
        void increase_skill(size_t idx,double inc){
            set_skill(idx,skill_level_[idx]+inc);
        };
        void set_skill(size_t idx,double val){
            double threashold(0.05);
            size_t idn(0);
            assert(idx<FOOD_TYPES);
            assert(FOOD_TYPES==2);
            if(idx==0) idn=1;
            if(idx==1) idn=0;
            skill_level_[idx]=val;
            if(skill_level_[idx]>(1.-threashold)) skill_level_[idx]=(1.-threashold);
            if(skill_level_[idx]<threashold) skill_level_[idx]=threashold;
            skill_level_[idn]=1.-skill_level_[idx];
            assert(skill_level_[idx]>0);
            assert(skill_level_[idx]<1);
            assert(skill_level_[idn]<1);
            assert(skill_level_[idn]>0);
        };
        Genome::actions_type test_input(const Genome::perception_type input)const; // dummy function, TODO remove?
        //! Increments the weights
        /*!
          Increases the weights between the input and the action by val
          Used in the simulation to generate a subpopulation of herding agents
         */
        void seed(Genome::perception_type input,int action,double val);
        void fill_weights_random();

        // This method lets cereal know which data members to serialize
        template<class Archive>
        void serialize(Archive & archive)
        {
            archive(CEREAL_NVP(brain_),CEREAL_NVP(skill_level_)); // serialize things by passing them to the archive
        }



    };

    const struct constants_t
    {
        constants_t(){}         // llvm complains if this is missing

#ifndef INTERACT
#ifndef SEP_FOOD
        const static size_t len=5;
        const std::vector<std::string> names = {"foodH", "foodN", "foodW", "foodE", "foodS"};
        const std::vector<Genome::perception_type> tests = {
            {MAX_INPUT_VAL,0,0,0,0},
            {0,MAX_INPUT_VAL,0,0,0},
            {0,0,MAX_INPUT_VAL,0,0},
            {0,0,0,MAX_INPUT_VAL,0},
            {0,0,0,0,MAX_INPUT_VAL}
        };
#else
        const static size_t len=6;
        const std::vector<std::string> names = {"foodH0","foodH1","foodN", "foodW", "foodE", "foodS"};
        const std::vector<Genome::perception_type> tests = {
            {MAX_INPUT_VAL,0,0,0,0,0},
            {0,MAX_INPUT_VAL,0,0,0,0},
            {0,0,MAX_INPUT_VAL,0,0,0},
            {0,0,0,MAX_INPUT_VAL,0,0},
            {0,0,0,0,MAX_INPUT_VAL,0},
            {0,0,0,0,0,MAX_INPUT_VAL}
        };
#endif

#elif defined INTERACT
#ifdef INVISIBLE_FOOD

#ifndef SEP_FOOD
        const static size_t len=5;
        const std::vector<std::string> names = {"foodH", "agentN", "agentW","agentE", "agentS"};
        const std::vector<Genome::perception_type> tests = {
            {MAX_INPUT_VAL,0,0,0,0},
            {0,MAX_INPUT_VAL,0,0,0},
            {0,0,MAX_INPUT_VAL,0,0},
            {0,0,0,MAX_INPUT_VAL,0},
            {0,0,0,0,MAX_INPUT_VAL}
        };
#else
        const static size_t len=7;
        const std::vector<std::string> names = {"foodH0","foodH1","agentH","agentN", "agentW", "agentE", "agentS"};
        const std::vector<Genome::perception_type> tests = {
            {MAX_INPUT_VAL,0,0,0,0,0,0},
            {0,MAX_INPUT_VAL,0,0,0,0,0},
            {0,0,MAX_INPUT_VAL,0,0,0,0},
            {0,0,0,MAX_INPUT_VAL,0,0,0},
            {0,0,0,0,MAX_INPUT_VAL,0,0},
            {0,0,0,0,0,MAX_INPUT_VAL,0},
            {0,0,0,0,0,0,MAX_INPUT_VAL}
        };
#endif
#elif !defined INVISIBLE_FOOD

#ifndef SEP_FOOD
        const static size_t len=10;
        const std::vector<std::string> names = {"foodH", "agentH", "foodN", "foodW","foodE", "foodS", "agentN", "agentW", "agentE", "agentS"};
        const std::vector<Genome::perception_type> tests = {
            {MAX_INPUT_VAL,0,0,0,0,0,0,0,0,0},
            {0,MAX_INPUT_VAL,0,0,0,0,0,0,0,0},
            {0,0,MAX_INPUT_VAL,0,0,0,0,0,0,0},
            {0,0,0,MAX_INPUT_VAL,0,0,0,0,0,0},
            {0,0,0,0,MAX_INPUT_VAL,0,0,0,0,0},
            {0,0,0,0,0,MAX_INPUT_VAL,0,0,0,0},
            {0,0,0,0,0,0,MAX_INPUT_VAL,0,0,0},
            {0,0,0,0,0,0,0,MAX_INPUT_VAL,0,0},
            {0,0,0,0,0,0,0,0,MAX_INPUT_VAL,0},
            {0,0,0,0,0,0,0,0,0,MAX_INPUT_VAL}
        };
#else
        const static size_t len=11;
        const std::vector<std::string> names = {"foodH0","foodH1", "agentH","foodN", "foodW","foodE", "foodS", "agentN", "agentW","agentE", "agentS"};
        const std::vector<Genome::perception_type> tests = {
            {MAX_INPUT_VAL,0,0,0,0,0,0,0,0,0,0},
            {0,MAX_INPUT_VAL,0,0,0,0,0,0,0,0,0},
            {0,0,MAX_INPUT_VAL,0,0,0,0,0,0,0,0},
            {0,0,0,MAX_INPUT_VAL,0,0,0,0,0,0,0},
            {0,0,0,0,MAX_INPUT_VAL,0,0,0,0,0,0},
            {0,0,0,0,0,MAX_INPUT_VAL,0,0,0,0,0},
            {0,0,0,0,0,0,MAX_INPUT_VAL,0,0,0,0},
            {0,0,0,0,0,0,0,MAX_INPUT_VAL,0,0,0},
            {0,0,0,0,0,0,0,0,MAX_INPUT_VAL,0,0},
            {0,0,0,0,0,0,0,0,0,MAX_INPUT_VAL,0},
            {0,0,0,0,0,0,0,0,0,0,MAX_INPUT_VAL}
        };
#endif
#endif //INVISIBLE_FOOD
#endif //INTERACT
    } visualsys;

} // end namespace Joleste

#endif // !defined GENOME_HPP
