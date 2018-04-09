#include "agent.hpp"
#include <cassert>
#ifdef DEBUG
#include <iostream>
#endif //DEBUG

namespace Joleste
{
    age_type Agent::maximum_age_=100;
    //! Default costructor.
    /*!
      Initializes the energy at a random value computed adding a random value between 0 and INIT_ENERGY_VAR to the constant value INIT_ENERGY_CONST
      It then mutates the genome.

      \return An Agent object
    */

    //Replicate constructor
    Agent::Agent( const Genome &gen, /*!< The genome to give to the agent */
                  int ID  /*!< The unique identifier of the agent */
                  )             //phenotype - inefficient creating an object that will be replaced better phenotype_(genotype_.mutate() )//TODO?
        : age_(0),to_delete(false),genotype_(gen,ID),phenotype_(gen,ID), ID_(ID),last_feedback(0),seed_(0),action(-1)
    {
        std::uniform_int_distribution<int> distribution_for_energy(1,INIT_ENERGY_VAR);
        energy_=INIT_ENERGY_CONST+distribution_for_energy(rng);
        genotype_.mutate(); //Should mutate better return a reference used to instantiate phenotype?
        phenotype_=genotype_; //Inefficient ^ see above
    } // Constructor using a genome as input (should either be fct or later the neural network
    //! Take a simulation step
    /*!
      - Check if the agent is dead
      - increase its age
      - Preprocess perceptions
      - Take a decision

      \returns A number representing the action taken
    */
    int Agent::choose_action(const Genome::perception_type neighbors, /*!< The perceptions vector of the current state */
                    const bool binary_in, /*!< if true convert perception vector to binary before sending it to the brain  */
                    const int &timestep
                    )const//,Genome::marker_type avg_markers,Genome::marker_type prey_markers)
    {
       Genome::perception_type neighbors_input(neighbors);
#ifdef DEBUG
        if(is_dead())
            std::cout<<"Agent "<<ID_<<" is dead! energy "<<energy_<<" age "<<age_<<std::endl;
#endif //DEBUG
        if(is_dead())             // in case some other agent killed it previously this turn
            return -1;            // default case in population.
#ifndef INVISIBLE_FOOD
        assert(neighbors.size()==N_PERCEPTIONS);
#endif //INVISIBLE_FOOD
        // ---------- increase age ----------
        // ---------- update perceptions ----------
        for (size_t i = 0; i < neighbors_input.size(); i++){ // convert food and agents quantities in binary value (0 or MAX_INPUT_VAL)
            if(neighbors[i]!=0) {
                if(binary_in)
                    neighbors_input[i]=MAX_INPUT_VAL;
                else
                    //neighbors_input[i]=ENERGY_MAX*neighbors[i]/(neighbors[i]+1);
                    neighbors_input[i]=std::min(MAX_INPUT_VAL*1.0,neighbors_input[i]);
            }
        }
        // ---------- take decision ----------
        return decide_action(neighbors_input,timestep);
    }

    //! Decide which action to take.
    /*!
      - Train the brain with the reward of the last timestep
      - Receive from genome the decision vector
      - Add some noise to the decision, to break ties
      - Select and return one action (the one with the highest value)

      \returns A number representing the action taken
     */
    int Agent::decide_action(const Genome::perception_type inputs,const int &timestep) const// here the genome (function or nn) should come into action
    {
        assert(inputs.size()==N_PERCEPTIONS);
#ifdef DEBUG
        std::cout<<std::endl<<"---------- Agent "<<get_ID()<<" plays ----------"<<std::endl;
#endif
#ifdef LEARN
        //Needs mutex if conccurent
        phenotype_.train(inputs,last_feedback,timestep);
#endif
#ifdef DEBUG
        std::cout<<"^^^^^^^^ Agent "<<get_ID()<<" ENDS ^^^^^^^^^^^^^^^^^^^^^^"<<std::endl<<std::endl;
#endif
        //Needs mutex if conccurent
        last_feedback=0;          // will be updated by the next feedback
        return phenotype_.decision_algorithm(inputs);
    }

    //! Return if the agent is dead.
    /*!
      If the compile flag IMMORTALS is enabled, this always returns false.
      Else check if the energy went below 0 or the age above the maximum age

      \returns A boolean indicating whether the agent is dead
     */
    bool Agent::is_dead() const // return if agent is dead or alive
    {
        return
#ifdef IMMORTALS
            false;
#elif !defined IMMORTALS
        (
         (energy_ <= 0 ? true : false) ||
         (age_    >= maximum_age_ ? true : false) // TODO
         );
#endif
    }


    //! Convert a number to the corresponding action string
    /*!
      \param num An integer
      \returns A string
     */
    std::string Agent::ntoa(int num)
    {
        switch(num){
        case 0: return "north";
        case 1: return "west";
        case 2: return "east";
        case 3: return "south";
        case 4: return "eat";
        default: return "unknown";
        }
    }

    //! Convert an action string to the corresponding number
    /*!
      \param act A string
      \returns An integer
    */
    double Agent::aton(std::string act)
    {
        if(act=="north")
            return 0;
        else if(act=="west")
            return 1;
        else if(act=="east")
            return 2;
        else if(act=="south")
            return 3;
        else if(act=="eat")
            return 4;
        else
            return -1;
    }

    //! Select an action for the current input, without training the brain.
    /*!
      \param input A vector of perceptions
      \returns An integer, the value of the chosen action
     */
    size_t Agent::test_configuration(const Genome::perception_type input, bool phen) const
    {
        Genome::perception_type result(N_OUTPUTS);
        std::vector<size_t> idx(N_OUTPUTS);
        // initialize original index locations
        for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
        if(phen){
            result = phenotype_.test_input(input);
        }else{
            result = genotype_.test_input(input);
        }
        // sort indexes based on comparing values in v
        sort(idx.begin(), idx.end(),
             [&result](size_t i1, size_t i2) {return result[i1] > result[i2];}); // TODO move this to a new function "decision_algorithm"
#ifdef DEBUG
        for(auto &a:idx)
            std::cout<<ntoa(a)<<",";
        std::cout<<"\twith values: ";
        for(auto &r:idx)
            std::cout<<result[r]<<"|";
        std::cout<<std::endl;
#endif
        return idx[0];
    }

    //! Test the behavior of an agent.
    /*!
      Test the agent on different situations, based on the perceptions it receives
      \returns A vector with the actions chosen for each test

      When changing this function, change as well the order of the column names in Population::simulate
     */
    Genome::actions_type Agent::test_agent(bool phen) const
    {
#ifdef DEBUG
        std::cout<<"--------------------------------------------------"<<std::endl;
        std::cout<<"Testing agent "<<ID_<<std::endl;
#endif
        std::vector<Genome::perception_type> inputs=visualsys.tests;
        Genome::actions_type result;
        for(auto&t:inputs){
            result.push_back(test_configuration(t,phen));
        }

#ifdef DEBUG
        //std::cout<<"|----QTABLE----| ID "<<this->ID_<<std::endl;
        //phenotype_.brain_.print_Qtable();
        //std::cout<<"|--------------|"<<std::endl;

        //std::cout<<"--------------------------------------------------"<<std::endl;
#endif
        return result;
    }

    void Agent::seed_foraging(double val){
        std::cout<<"Seeding foraging of agent "<<ID_<<std::endl;
        Genome::perception_type input(N_PERCEPTIONS);
#ifndef SEP_FOOD
        input={MAX_INPUT_VAL,0,0,0,0};
        phenotype_.seed(input,aton("eat"),val);
        genotype_.seed(input,aton("eat"),val);
#else
        input={MAX_INPUT_VAL,0,0,0,0,0}; // food0
        phenotype_.seed(input,aton("eat"),val);
        genotype_.seed(input,aton("eat"),val);
        input={0,MAX_INPUT_VAL,0,0,0,0}; // food1
        phenotype_.seed(input,aton("eat"),val);
        genotype_.seed(input,aton("eat"),val);
#endif
    }

    //! Seed the agent with social behavior
    /*!
      Force the agent to follow the others
     */
    void Agent::seed_social(double val /*!< How much to increment the weight */){
#ifdef INTERACT
#ifdef INVISIBLE_FOOD
        seed_=1;
        std::cout<<"Seeding agent "<<ID_<<" socially"<<std::endl;
        Genome::perception_type input(N_PERCEPTIONS);
        // gen_.fill_weights_random();
        //food here
        // input={0,0,MAX_INPUT_VAL,0,0};
        // gen_.seed(input,aton("eat"),val);
        // --------------------------------------------------
#ifndef SEP_FOOD
        //no food, agent north:\t";
        input={0,MAX_INPUT_VAL,0,0,0};
        phenotype_.seed(input,aton("north"),val);
        // --------------------------------------------------
        //no food, agent west
        input={0,0,MAX_INPUT_VAL,0,0};
        phenotype_.seed(input,aton("west"),val);
        // --------------------------------------------------
        //no food, agent east
        input={0,0,0,MAX_INPUT_VAL,0};
        phenotype_.seed(input,aton("east"),val);
        // --------------------------------------------------
        //no food, agent south
        input={0,0,0,0,MAX_INPUT_VAL};
        phenotype_.seed(input,aton("south"),val);
        //gen_.mutate();
#else
        //no food, agent north:\t";
        input={0,0,MAX_INPUT_VAL,0,0,0};
        phenotype_.seed(input,aton("north"),val);
        // --------------------------------------------------
        //no food, agent west
        input={0,0,0,MAX_INPUT_VAL,0,0};
        phenotype_.seed(input,aton("west"),val);
        // --------------------------------------------------
        //no food, agent east
        input={0,0,0,0,MAX_INPUT_VAL,0};
        phenotype_.seed(input,aton("east"),val);
        // --------------------------------------------------
        //no food, agent south
        input={0,0,0,0,0,MAX_INPUT_VAL};
        phenotype_.seed(input,aton("south"),val);
        //gen_.mutate();
#endif
#endif
#endif
    }
    //! Seed the agent with antisocial behavior
    /*!
      Force the agent to run away from the others
    */
    void Agent::seed_antisocial(double val /*!< How much to increment the weight */){
#ifdef INTERACT
#ifdef INVISIBLE_FOOD
        seed_=2;
        std::cout<<"Seeding agent "<<ID_<<" antisocially"<<std::endl;
        Genome::perception_type input(N_PERCEPTIONS);
        //no food, agent here
        //input={1,0,0,MAX_INPUT_VAL,0,0};
        //gen_.seed(input,5,val);     // eat
        // --------------------------------------------------
#ifndef SEP_FOOD
        //no food, agent north:\t";
        input={0,MAX_INPUT_VAL,0,0,0};
        phenotype_.seed(input,aton("south"),val);
        // --------------------------------------------------
        //no food, agent west
        input={0,0,MAX_INPUT_VAL,0,0};
        phenotype_.seed(input,aton("east"),val);
        // --------------------------------------------------
        //no food, agent east
        input={0,0,0,MAX_INPUT_VAL,0};
        phenotype_.seed(input,aton("west"),val);
        // --------------------------------------------------
        //no food, agent south
        input={0,0,0,0,MAX_INPUT_VAL};
        phenotype_.seed(input,aton("north"),val);
        //gen_.mutate();
#else
        //no food, agent north:\t";
        input={0,0,MAX_INPUT_VAL,0,0,0};
        phenotype_.seed(input,aton("south"),val);
        // --------------------------------------------------
        //no food, agent west
        input={0,0,0,MAX_INPUT_VAL,0,0};
        phenotype_.seed(input,aton("east"),val);
        // --------------------------------------------------
        //no food, agent east
        input={0,0,0,0,MAX_INPUT_VAL,0};
        phenotype_.seed(input,aton("west"),val);
        // --------------------------------------------------
        //no food, agent south
        input={0,0,0,0,0,MAX_INPUT_VAL};
        phenotype_.seed(input,aton("north"),val);
        //gen_.mutate();
#endif
#endif
#endif
    }

} // end namespace Joleste
