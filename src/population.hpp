#ifndef POPULATION_HPP
#define POPULATION_HPP

#include "agent.hpp"
#include "field.hpp"
#include "constants.h"
#include <list> ///operations on list most efficient for high population sizes!
#include <vector>               // but list does not support accessing elements by their index
/// we recently made the test for the different containers (vector, list, set). See the results in the folder (containers.png)
// TODO from the graph list seems to be the slowest...
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <cassert>
#include <utility>            // needed for swap
#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdlib.h>
#include <fstream>
#include <string>


#define assert__(x) for ( ; !(x) ; assert(x) )

// determines how many agents are attempted to be killed during every round. If the value is 0, the whole population is extracted.
#define FRACTION_DEATHS 0.0
#define LOG_FREQ 10

namespace Joleste
{
    //! Class population.
    /*!
      This is the main class of the simulation.
      It contains:
      - the population (vector of agents)
      - the environment (vector of cells)
    */

    class Population
    {
        typedef Field field_type; //! containing field and number of agents in that field
        typedef std::pair<Agent, int> population_type; /*!< This type associates an agent to the id of the cell it is in */
        typedef std::vector<population_type>::const_iterator const_iterator; /// create iterator for the population


        // variables
        int popID_; //Saves the ID of the population object
        std::size_t nmax_;      /*!< max number of agents */
        std::size_t nmin_;      /*!< min number of agents */
        std::size_t total_num_food_cells_;  /*!< total number of food sources */
        double food_proportion_;
        std::size_t season_length_;
        std::size_t num_food_cells[FOOD_TYPES];
        std::vector<population_type> population_; /*!< The population */
        //size_t population_size_;
        std::vector<field_type> fields_; /*!< The environment */
        int agent_counter_;     /*!< defines the unique ID of a new agent, needs to be incremented any time an agent is created */
        size_t food_max_; /*!< The maximum quantity of food a food source can contain */
        size_t fov_radius_; /*!< The range of view of agents */
        bool binary_in_; /*!< Whether agents perceive continuous or binary values */
        bool direct_feedback_; /*!< If enabled agents get a reward when entering a cell with food. Useful for learning */
        std::size_t max_age_;   // the maximum life expectancy
        std::size_t n0_;        // initial population size
        size_t field_size_; /*!< The size of the grid */
        double prob_eat_success_[FOOD_TYPES]; /*!< fail to forage with a given probability */
        double food_energy_; /*!< receive a penalty if foraging fails */
        double life_bonus_;  /*!< how much reproduction is favored over death, higher means more reproduction */
        uint seed_iteration_;
        uint famine_iteration_;
        std::vector<size_t> save_pop_;
        double social_ratio_=0;
        double antisocial_ratio_=0;

        //std::ofstream logFile;
        std::ofstream statsFileAgent; /*!< The output file */
        std::ofstream statsFileAgent_gen; /*!< The output file */
        std::ofstream statsFileEnv; /*!< The output file */
        std::ofstream statsFileReprod; /*!< The output file */
        std::ofstream statsFileForage; /*!< The output file */

    public:

  //! Constructor
        /*!
\param [in] nmax Max number of agents, start killing agents any time this threshold is passed
        \param [in] nmin Min number of agents, spawn random agents anytimes population is smaller than this value
            \param [in] n0 Initial number of agents
            \param [in] f0 Total number of food sources, can change during the simulation
            \param [in] fmax The maximum quantity of food a food source can contain
            \param [in] field_size The size of the grid, X*X
            \param [in] max_age The maximum age an agent can reach
            \param [in] fov_radius The range of view of agents
            \param [in] binary_in Whether agents perceive continuous or binary values
            \param [in] direct_feedback If enabled agents get a reward when entering a cell with food. Useful for learning
            \param [in] probability of successfully eating food 0
            \param [in] food_kill_energy Receive a penalty if foraging fails

            \returns A Population object
         */
        Population(int popID,std::size_t nmax,std::size_t nmin,std::size_t n0,std::size_t f0,double food_proportion,std::size_t season_length,float mutation_size_skill_,float mutation_size_weights_,std::size_t fmax,std::size_t field_size,age_type max_age,size_t fov_radius,bool binary_in,bool direct_feedback,double prob_eat_success_0,double food_energy,double life_bonus,int seed_iteration,int famine_iteration,std::vector<size_t> save_pop,std::string load_pop,std::string load_logic);

        // Classes with a vtable should have a virtual destructor.
  // just to remember vtable contains pointers to the virtual functions! Good to remember for an job interview :-)
  // There can only be one vtable per class, and all objects of the same class will share the same vtable. You need them!
        virtual ~Population();

  //! Simulate.
        /*!
          \param [in] time How many timesteps to simulate
          \param [in] social_ratio The percentage of population to seed with social behavior
          \param [in] antisocial_ratio The percentage of population to seed with antisocial behavior
          \param [in] fileName The name of the output file, if empty no output
         */
        void simulate(std::vector<float> &res_phen,std::vector<float> &res_SDphen,std::vector<float> &res_gen,std::vector<float> &res_SDgen,std::size_t time,double social_ratio=0,double antisocial_ratio=0,const std::string &fileNameAgents = std::string(),const std::string &fileNameAgents_gen = std::string(),const std::string &fileNameEnv = std::string(),const std::string &fileNameReprod=std::string(),const std::string &fileNameForage=std::string(),const std::string &fileNamePop=std::string());
        std::size_t size() const {return population_.size();} /// Get size of population

  /*
    function to update the food stored in the vector (private) of cell int
    could be done globally in the function (randomly for all the cells!
    this would seem more natural to me
    void spawn_bundle(int);
    also remove (eat) bundle in cell int
    void consume_food(int, int);
    TODO moved to class Field
  */

  /// Simulate one time step (year).
  /// this contains: shuffle agents, check for neighbours (population_.second), give each agent the parameters required,
  /// execute the agents functions (make decision,...) and take the results (depending of action) to change agents position/fight, etc.
  /// also check pregnancy and if_dead and take action
  //virtual std::pair<std::string,std::string> step(std::size_t timeStep);
    virtual void step(const std::size_t timeStep, std::vector<float> &result_phen,std::vector<float> &result_gen,bool &failed);


    private:
        //! These methods return the pointer to the agent for different input types
        Agent& get_agent_at(population_type &a){return a.first;}
        const Agent& get_agent_at(const population_type &a)const{return a.first;}
        Agent& get_agent_at(const int index){return population_[index].first;}
        Agent& get_agent_at(const std::vector<population_type>::iterator it){return (*it).first;}
        //! These methods return the pointer to the field for different input types
        int get_field_at(population_type &a) {return a.second;}
        int get_field_at(const population_type &a) const{return a.second;}
        int get_field_at(const int index)const {return population_[index].second;}
        int get_field_at(const const_iterator it)const {return (*it).second;}
  // functions
        std::vector<int> get_neighboring_cells(const int,const int,const int,const int) const; /*!< returns the ids of the cells in the neighborhood of the current cell */
        std::vector<std::vector<int> > field_of_view(const int,const int,const int,const int) const; /*!< returns the indexes of the cells in the field of view, divided by location */
        void del_agent(std::vector<population_type>::iterator &it); /*!< delete an agent from the population */
        void del_agent(int i);
        Genome::perception_type compute_agent_perceptions(const population_type &a,const int fieldID) const; /*!< computes the perception of an agent */
        Genome::perception_type compute_agent_perceptions2(const population_type &a,const int fieldID) const; /*!< computes the perception of an agent */
        void agent_moves(population_type &a,int orientation); /*!< The agent decides to move to another cell */
        void agent_eats(population_type &a,size_t &num_food,size_t (&by_type_cells_w_food)[FOOD_TYPES], const size_t timeStep,  std::ostringstream &str_stats); /*!< The agent decides to eat */
        bool agent_tries_to_eat(population_type &a,size_t &num_food,size_t (&by_type_cells_w_food)[FOOD_TYPES],size_t food_type,bool always_eat);
        Agent agent_replicates(Agent &a); /*!< The agent replicates */
        bool is_agent_successful(const population_type &a,const size_t &food_type)const;
        void shuffle_agents();  // shuffles the population
        void shuffle_agents_pointers(std::vector<population_type*> &ppopulation);
        void initialize_fields();
        void initial_spawn_food();
        void initialize_population(std::string load_pop,std::string _enotype,float mutation_size_skill_,float mutation_size_weigths_);
        void reproduction(size_t timeStep,std::ostringstream &Reprod_str_stats,double life_bonus_); /*!< All agents that can will reproduce */
        void replenish_population(); /*!< Spawn more agents if the population is too small (below nmin_) */
        void replenish_food(int tStep,size_t &num_food,size_t (&num_food_bt)[FOOD_TYPES]);
        void set_food_quantities(int day_of_season,int season_num);
        void remove_dead_agents();
        void cap_population();
        void debug_step(int fieldID, Genome::perception_type perceptions); //TODO unused
        void seed_population();
        void create_food_source(std::vector<field_type> &fields,int x0,int x1,int init_food,size_t food_type, bool exclusive, bool locked);
        void remove_food_source(std::vector<field_type> &fields,size_t food_type);


        void count_and_lock_food_fields(size_t &cells_w_food,size_t (&by_type_cells_w_food)[FOOD_TYPES],const size_t &timeStep, bool lock);

        //IO functions
        void log_INIT(std::ofstream &statsFile,const std::string &fileName,std::vector<std::string> &colnames);
        void log_WRITE(std::ostringstream &str_stats,std::ofstream &statsFile);
        void log_env_stats_INIT(const std::string &fileNameEnv);
        void log_env_stats(std::ostringstream &str_stats,const size_t &timeStep);
        void log_agent_stats_INIT(const std::string &fileNameAgents,std::ofstream &statsFile);
        void log_agent_stats(const Agent &agent, const int fieldID,const Genome::actions_type &result,const size_t &timeStep,std::ostringstream &str_stats)const;
        void log_reprod_stats_INIT(const std::string &fileName);
        void log_reprod_stats(Agent &agent,Agent &parent, const size_t &timeStep, std::ostringstream &str_stats);
        void log_forage_stats_INIT(const std::string &fileNameEnv);
        void log_forage_stats(Agent &agent, const size_t &timeStep, const bool success, const size_t food_type, std::ostringstream &str_stats);

    public:
        void save_population(std::string filename);
        void load_population(std::string filename);

    private:
        //Visualization functions
        void output_grid();
        void output_food_fields(int tStep);
        void output_agents(int tStep);

        //! The following methods print nicely stats about the population
        std::string print_pop() {return print_pop(population_);}
        std::string print_pop(int seed);
        std::string print_pop(std::vector<population_type> pop);
        //! The following methods return stats about the population
        std::vector<std::vector<double> > write_pop() {return write_pop(population_);}
        void write_pop(int seed);
        std::vector<std::vector<double> > write_pop(std::vector<population_type> pop);

        int count_food(const Field field,const bool (&see_successful)[FOOD_TYPES]) const;
        int count_agents(const Field field,const bool same_field) const;
        int count_food_TYPE(const int food_type,const Field field,const bool (&see_successful)[FOOD_TYPES]) const;

        //! Get iterators for the population_ container
        const_iterator begin() const {return population_.begin();}
        const_iterator end() const {return population_.end();}



    };

} // end namespace Joleste

#endif // !defined POPULATION_HPP
