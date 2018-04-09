#ifndef FIELD_HPP
#define FIELD_HPP

#include "genome.hpp"
#include <algorithm>
#include <functional>
#include <cassert>
#include "constants.h"


namespace Joleste
{
    //!  Field class
    /*!
      This class implements a cell in the environment.
      This class is used to keep track of the food in the environment and to compute agents' perceptions
    */

    class Field
    {
    private:
        int food_[FOOD_TYPES];                   /*!< amount of food */
        int food_unlocked_[FOOD_TYPES];                   /*!< food field is locked */
        int num_agents_;             /*!< number of agents in the cell */
        int initial_food_[FOOD_TYPES];           /*!< Initial quantity of food that has been spawned */
        int times_unlocked[FOOD_TYPES]; //Should always be 0 or 1
        int times_shared[FOOD_TYPES]; //Times eaten when it was already unlocked
    public:
        //! Default constructor: the cell starts without food nor agents
        Field():num_agents_(0){
            for(int i=0;i<FOOD_TYPES;i++){
                food_[i]=0;
                food_unlocked_[i]=true;
                initial_food_[i]=0;
                times_unlocked[i]=0;
                times_shared[i]=0;
            }
            // ---------- initialize average markers ----------
            //avg_markers_.resize(N_MARKERS,0.0); // an empty marker initialized to 0. Initially all agents have the same markers
        }
        void add_agent(){num_agents_++;};   /*!< increments of one the number of agents in this cell */
        void rem_agent(){                   /*!< decrements of one the number of agents in this cell */
            if(num_agents_>0)
                num_agents_--;
            else
                num_agents_=0;
        };
        //! Execute a simulation step.
        /*! At each step there is a small probability of increasing the food in this cell
          Is this is ever used?
        */

        //For IO
        void reset_times_unlocked(size_t idx){assert(idx<FOOD_TYPES);times_unlocked[idx]=0;}
        int get_times_unlocked(const size_t idx)const{assert(idx<FOOD_TYPES);return times_unlocked[idx];}
        void inc_times_unlocked(size_t idx){assert(idx<FOOD_TYPES);times_unlocked[idx]++;}
        void reset_times_shared(size_t idx){assert(idx<FOOD_TYPES);times_shared[idx]=0;}
        int get_times_shared(const size_t idx)const{assert(idx<FOOD_TYPES);return times_shared[idx];}
        void inc_times_shared(size_t idx){assert(idx<FOOD_TYPES);times_shared[idx]++;}

        int get_num_agents() const{return num_agents_;} /*!< returns the number of agents in this cell */
        int get_food(const size_t idx)const {assert(idx<FOOD_TYPES);return food_[idx];} /*!< returns the number of food units in this cell */
        bool is_food_unlocked(const size_t idx)const{assert(idx<FOOD_TYPES);return food_unlocked_[idx];}
        void lock_food(size_t idx){assert(idx<FOOD_TYPES);food_unlocked_[idx]=false;}
        void unlock_food(size_t idx){assert(idx<FOOD_TYPES);food_unlocked_[idx]=true;}
        int get_all_food() const {int total(0);for(size_t idx=0;idx<FOOD_TYPES;idx++) total+=food_[idx];return total;} /*!< returns the number of food units in this cell */
        int get_all_unlocked_food()const{int total(0);for(size_t idx=0;idx<FOOD_TYPES;idx++){ if(food_unlocked_[idx]) total+=food_[idx];} return total;} /*!< returns the number of food units in this cell */
        //! Set the initial food in the cell
        /*! Initial food is used to keep track of how many cells in the environment contain food
          This helps distinguishing cells that ran out of food from cells that never had any
          It is used to keep the number of food sources constant during the simulation
        */
        void set_initial_food(size_t idx,int i){assert(idx<FOOD_TYPES);initial_food_[idx]=i;}
        int get_initial_food(const size_t idx)const{assert(idx<FOOD_TYPES);return initial_food_[idx];}
        //! Increment quantity of food
        void inc_food(size_t idx){assert(idx<FOOD_TYPES);food_[idx]++;}
        void inc_food(size_t idx,int n){assert(idx<FOOD_TYPES);food_[idx]+=n;}
        void set_food(size_t idx,int n){assert(idx<FOOD_TYPES);food_[idx]=n;}
        //! Decrement food of one unit
        /*! Called any time an agent eats */
        void consume_bundle(size_t idx) {
            assert(idx<FOOD_TYPES);
            if(food_[idx]>0)
                food_[idx]--;
            else
                food_[idx]=0;
        };
    private:
        //! Increases the food of one with a certain probability
        /*! Is this is ever used?
         */
        void spawn_bundle() {
            //Spawns all types of food
            for(size_t idx=0;idx<FOOD_TYPES;idx++){
                std::uniform_real_distribution<double> food_dist(0.0,1.0);
                if(food_dist(rng)<GRASS_SPAWN_PROB)
                    food_[idx]++;                // TODO is food discrete or continuous?
            }
        };
        void update_avg(double val,double &avg,int N,bool add); /*!< Was used to keep track of the average marker of the agents in this cell */
    };
} // end namespace Joleste

#endif // !defined POPULATION_HPP
