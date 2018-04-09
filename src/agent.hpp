#ifndef AGENT_HPP
#define AGENT_HPP

#include "genome.hpp"
#include <algorithm>
#include <iostream>
#include <math.h>
#include "constants.h"

//// Joleste is the declared namespace for the implementation (to make all things accessible in the implementation also in the other codes!)
//// it should be mentioned that Joleste represents three great guys on their challenge for the nobel... you may change the name. But please in all files then :-)
/// You could do so by: sed s/Joleste/<new_name>/g *.cpp; sed s/Joleste/<new_name>/g *.hpp
namespace Joleste
{
    //!  The class agent.
    /*!
      This class contains the agents.
      Each agent has a genome that drives its actions, which are movement, foraging, reproduction and death.
      The class also contains methods for testing or changing the behavior of the agent.
    */
    class Agent
    {
    public:
        //// static variables / functions ////
        static age_type maximum_age_; /*!< Agents die whenever above this age */
        static void alter_max_age(age_type ag){ maximum_age_ = ag; }

        public:
            age_type age_;
            bool to_delete;
        private:
            double energy_; /*!< The energy (life or score) of the agent */
            //Mutable objects need mutex for safe conccurency
            Genome genotype_; /*!< The genome (brain) of the agent (can be a RL algorithm or a neural net) */
            mutable Genome phenotype_; /*!< The genome (brain) of the agent ACTIVE (can be a RL algorithm or a neural net) */
            int ID_; /*!< The unique identifier of this agent*/
            mutable double last_feedback;/*!< Used to store the last reward, needed for learning */
            // TODO rename to last_reward
            bool const_mutation_rate_;
            int seed_;                /*!< Contains the id of the seed, used to track agents that got initialized with a specific behavior */
            mutable int action;


    public:
        //! Default Constructor.
        /*!
         Should be used for serialization purposes only, creates an empty agent
        */
        Agent(){
#ifdef DEBUG
            std::cerr<<" This default constructor should be called ONLY during serialization"<<std::endl;
#endif
        };


        //! Costructor.
        /*!
          It generates a new random genome for this agent.
          Used to spawn a new agent.
        */
        Agent(int ID, /*!< The unique identifier of the agent */
              float mutation_size_skill_,
              float mutation_size_weigths_,
              double (&skill_level)[FOOD_TYPES]
              ) : Agent(Genome(skill_level,mutation_size_skill_,mutation_size_weigths_),ID) {
            //seed with instructions to eat
            // #ifdef INTERACT
            // #ifdef INVISIBLE_FOOD
            // Initialize genome to eat whenever there is food in the current cell
            //food here, no agents
            //Genome::perception_type input={0,0,MAX_INPUT_VAL,0,0};
            //gen_.seed(input,aton("eat"),MAX_WEIGHT);
            // #endif
            // #endif
        }

        //! Constructor.
        /*!
          Accepts a genome as parameter, which is then mutated.
          Used to replicate a new agent.
        */
        Agent( const Genome &gen, int ID);
        //Agent( const Agent &parent);
        int choose_action(const Genome::perception_type neighbors,const bool binary_in,const int &timestep) const;// ,Genome::marker_type avg_markers,Genome::marker_type prey);

        //// Setters & Getters ////
        int get_ID()const{return ID_;}
        void set_ID(int id){ID_=id;}
        int get_seed()const{return seed_;}
        int get_action()const{return action;}
        void set_action(int action_){action=action_;}
        age_type get_age() const {return age_;} //! return the age of the agent
        double get_energy()const {return energy_;} //! get energy of agent
        double get_temp(){return phenotype_.return_temp();}
        double get_skill(const size_t idx)const {return phenotype_.return_skill(idx);}
        double get_skill_gen(const size_t idx)const {return genotype_.return_skill(idx);}
        void set_skill(size_t idx,double val){phenotype_.set_skill(idx,val);}
        void use_genotype(){phenotype_=genotype_;}
        void use_phenotype(){genotype_=phenotype_;}
//        double get_weights(size_t i,size_t j){return genotype_.get_weights(i,j);}


        void set_seed(int s) {seed_=s;}
        void set_energy( double energy ) {energy_ = energy;} //! set energy of agent
        void vary_energy( double delta) {energy_ += delta; } //! Change the energy of the agent of a delta
        void birthday() {age_++;} //! let agent grow older by 1 year

        //// Actions ////
        void give_reward(double reward) {force_feedback(reward); vary_energy(reward);} /*!< Modifies the energy of the agent*/
        void force_feedback(double reward) {last_feedback=reward;} /*!< Modifies the energy of the agent*/
        void adapt_small(){phenotype_.mutate();} //! TODO unused?
        void increase_skill(size_t idx, double inc){phenotype_.increase_skill(idx,inc);}

        // we need to implement the decide-action in the population, so that the agent returns it's decision and then returns the decision in integer form
        // this we can then pick up in the population class and add the values health to the agent's health and substract it from the other agents health

        // here we should implement the basis function which accesses the (updated perceptions and returns an decision (int type)

        bool is_dead() const; /*!< return if agent is dead or alive */
        Agent replicate(int ID)  /*!< Create offspring inheriting its genome but with slightly random mutations */
        {
            //DISCUSSION NEEDED transmit learning or not
            //Agent ret(phenotype_,ID);
            Agent ret = Agent(genotype_,ID);
            ret.set_seed(this->get_seed());
            return ret;}
        Genome::actions_type test_agent(bool phen=true) const;

        //! Initialize the genome with a predefined value
        /*!
          This function sets the genome to one of predefined values
          Used in the simulation to generate a subpopulation of herding agents
        */
        void seed_foraging(double val);
        void seed_social(double val);
        void seed_antisocial(double val);

        static std::string ntoa(int num); /*!< converts an number to an action */
        static double aton(std::string act);/*!< converts an action to a number */
    private:
        size_t test_configuration(const Genome::perception_type input, bool phen=true)const;
        int decide_action(const Genome::perception_type inputs,const int &timestep) const; // here the genome (function or nn) should come into action

    public:
        template<class Archive>
        void save(Archive & archive) const
        {
            archive(CEREAL_NVP(genotype_),CEREAL_NVP(phenotype_)); // serialize things by passing them to the archive
        }

        template<class Archive>
        void load(Archive & archive)
        {
          archive(genotype_); // serialize things by passing them to the archive
          archive(phenotype_); // serialize things by passing them to the archive
          //Initialize other variables
          // remember to init ID somewhere else
          age_=0;
          to_delete=false;
          last_feedback=0;
          seed_=0;
          action=-1;
          std::uniform_int_distribution<int> distribution_for_energy(1,INIT_ENERGY_VAR);
          energy_=INIT_ENERGY_CONST+distribution_for_energy(rng);
        }
    };

} // end namespace Joleste

#endif
