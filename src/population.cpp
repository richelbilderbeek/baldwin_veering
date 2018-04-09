#include "population.hpp"
#include <cmath>

namespace Joleste
{

    //! Constructor
    /*!
      Initializes the field and the population
     */
    Population::Population( int popID,
                            size_t nmax, /*!< max number of agents, start killing agents any time this threshold is passed */
                            size_t nmin, /*!< min number of agents, spawn random agents anytimes population is smaller than this value */
                            size_t n0, /*!< initial number of agents */
                            size_t f0, /*!< total number of food sources, can change during the simulation */
                            double food_proportion,
                            size_t season_length,
                            float mutation_size_skill_,
                            float mutation_size_weigths_,
                            size_t fmax, /*!< The maximum quantity of food a food source can contain */
                            size_t field_size, /*!< The size of the grid, X*X */
                            age_type max_age, /*!< The maximum age an agent can reach */
                            size_t fov_radius, /*!< The range of view of agents */
                            bool binary_in, /*!< Whether agents perceive continuous or binary values */
                            bool direct_feedback, /*!< If enabled agents get a reward when entering a cell with food. Useful for learning */
                            double prob_eat_success_0, /*!< succed to forage food0 with a given probability */
                            double food_energy, /*!< receive a penalty if foraging fails */
                            double life_bonus, /*!< how much reproduction is favored over death, higher means more reproduction */
                            int seed_iteration, /*!<The population starts uniformly at random and at this timestep part of it gets seeded with one specific behavior */
                            int famine_iteration, /*!<The time at which the famine happens and resources are reduced */
                            std::vector<size_t> save_pop, /*!<The iterations at which to save the population */
                            std::string load_pop, /*!<The file name containin the population to load */
                            std::string load_logic /*!<The type of logic to use for loaded agents, either 'genotype' or 'phenotype'. If an empty string both genotype and phenotype are used. */
                            )
        :   popID_(popID)
        ,   nmax_( nmax )
        ,   nmin_( nmin )
        ,   total_num_food_cells_(f0)
        ,   food_proportion_(food_proportion)
        ,   season_length_(season_length)
        ,   agent_counter_(0) // The initial agent ID
        ,   food_max_(fmax)
        ,   fov_radius_(fov_radius)
        ,   binary_in_(binary_in)
        ,   direct_feedback_(direct_feedback)
        ,   max_age_(max_age)
        ,   n0_(n0)
        ,   field_size_(field_size) // size of grid, fields are ^2
        ,   food_energy_(food_energy)
        ,   life_bonus_(life_bonus)
        ,   seed_iteration_(seed_iteration)
        ,   famine_iteration_(famine_iteration)
        ,   save_pop_(save_pop)
    {
        Agent::alter_max_age(max_age);
        fields_.reserve(field_size_*field_size_);
        assert(f0<=field_size_*field_size_ && "cannot allocate more food cells than available cells");
        initialize_fields();         // set initial food
        initial_spawn_food(); //spawn food in the fields
        // ---------- initialize log file ----------
        //Following code needs to be changed with more than 2 food types
        assert(FOOD_TYPES==2);
        prob_eat_success_[0]=(prob_eat_success_0);
        prob_eat_success_[1]=(1-prob_eat_success_0);
        initialize_population(load_pop,load_logic,mutation_size_skill_,mutation_size_weigths_);   // generate initial population
        std::cout<<"done!"<<std::endl;
    }

    Population::~Population(){
        if(statsFileAgent.is_open())
            statsFileAgent.close();
        if(statsFileAgent_gen.is_open())
            statsFileAgent_gen.close();
        if(statsFileEnv.is_open())
            statsFileEnv.close();
        if(statsFileReprod.is_open())
            statsFileReprod.close();
        if(statsFileForage.is_open())
            statsFileForage.close();
    }

    //! Simulate.
    /*!
      It seeds the population and it writes output to file
     */
    void Population::simulate( std::vector<float> &res_phen,
                               std::vector<float> &res_SDphen,
                               std::vector<float> &res_gen,
                               std::vector<float> &res_SDgen,
                               size_t sim_length, //! How many timesteps to simulate
                               double social_ratio, //!The percentage of population to seed with social behavior
                               double antisocial_ratio,  /*!< The percentage of population to seed with antisocial behavior */
                               const std::string &fileNameAgents, /*!< The name of the output file, if empty no output */
                               const std::string &fileNameAgents_gen, /*!< The name of the output file, if empty no output */
                               const std::string &fileNameEnv, /*!< The name of the output file, if empty no output */
                               const std::string &fileNameReprod, /*!< The name of the output file, if empty no output */
                               const std::string &fileNameForage, /*!< The name of the output file, if empty no output */
                               const std::string &fileNamePop /*!< The name of the output file, if empty no output */
                             )
    {
        social_ratio_=social_ratio;
        antisocial_ratio_=antisocial_ratio;
        if(season_length_==0)
            season_length_=sim_length;

#ifdef VISUALIZE
        output_grid();
#endif
        // Initialize Logfiles
        log_env_stats_INIT(fileNameEnv);
#ifdef LEARN
        log_agent_stats_INIT(fileNameAgents,statsFileAgent);
#endif
        log_agent_stats_INIT(fileNameAgents_gen,statsFileAgent_gen);
        log_reprod_stats_INIT(fileNameReprod);
        log_forage_stats_INIT(fileNameForage);

        if(population_.size()>0){
        for (size_t i = 0; i < sim_length; i++) {
            if(std::find(save_pop_.begin(), save_pop_.end(), i) != save_pop_.end()){
                std::ostringstream filename;
                filename << fileNamePop<< i<<"_"<<popID_<<".xml";
                save_population(filename.str());
            }
// #ifdef INTERACT
// #ifdef INVISIBLE_FOOD
            if(i==famine_iteration_) {       // seed an agent with the social strategy
                //FAMINE CODE
//                std::cout<<"Seeding"<<std::endl;
//                total_num_food_cells_=5;
//                num_food_cells[0]=size_t((double) total_num_food_cells_ * food_proportion_);
//                num_food_cells[1]=total_num_food_cells_-num_food_cells[0];
            }
            if(i==seed_iteration_) {       // seed an agent with the social strategy
                std::cout<<"Seeding"<<std::endl;
                seed_population();
            }
// #endif
// #endif
            std::vector<float> result_phen;
            std::vector<float> result_gen;
            bool failed(false);
            step(i,result_phen,result_gen,failed);  // run and store a simulation step
            if(failed) return;

            //Ignore timesteps without results
            if(result_phen.size()){
                //Calculate results
                double total_phen(0.0);
                double total_gen(0.0);
                for(uint i=0;i<result_phen.size();i++){
                    total_phen+=result_phen[i];
                    total_gen+=result_gen[i];
                }
                double mean_phen(total_phen/(double)result_phen.size());
                double mean_gen(total_gen/(double)result_phen.size());
                res_phen.push_back(mean_phen);
                res_gen.push_back(mean_gen);

                total_phen=0.0;
                total_gen=0.0;
                double diff_phen(0.0),diff_gen(0.0);
                for(uint i=0;i<result_phen.size();i++){
                    diff_phen=(mean_phen-result_phen[i]);
                    diff_gen=(mean_gen-result_gen[i]);
                    total_phen+=(diff_phen*diff_phen);
                    total_gen+=(diff_gen*diff_gen);
                }
                total_phen=std::sqrt(total_phen/(double)result_phen.size());
                total_gen=std::sqrt(total_gen/(double)result_phen.size());
                res_SDphen.push_back(total_phen);
                res_SDgen.push_back(total_gen);
            }
        }
        }

        //All the logging functions should eventually move to an object
        if(statsFileAgent.is_open())
            statsFileAgent.close();
        if(statsFileAgent_gen.is_open())
            statsFileAgent_gen.close();
        if(statsFileEnv.is_open())
            statsFileEnv.close();
        if(statsFileReprod.is_open())
            statsFileReprod.close();
        if(statsFileForage.is_open())
            statsFileForage.close();
    }

    void Population::seed_population() {
        //// Initialize population with different initial values
        // TODO make it depend on whether the parameters (anti)social_ratio are set
#ifdef SEED_FORAGING
        for(size_t z=0;z<population_.size();z++){
            get_agent_at(z).seed_foraging(SEED_VALUE);
        }
#endif
#ifdef SEED_BEHAVIOR
        size_t social_start=0;
        size_t social_end=population_.size()*social_ratio_;
        size_t antisocial_start=social_end;
        size_t antisocial_end=social_end+population_.size()*antisocial_ratio_;

        for(size_t z=social_start;z<social_end;z++)
            get_agent_at(z).seed_social(SEED_VALUE);
        for(size_t z=antisocial_start;z<antisocial_end;z++)
            get_agent_at(z).seed_antisocial(SEED_VALUE);
#endif
    }

    //! Simulate one timestep.
    /*!
      - Logs the information about the fields with food
      - Generates offspring for agents that can reproduce
      - Spawns new agents if the population is too small
      - Shuffles the order of the agents
      - Executes one move for each agent
      - Tests and logs each agent's behavior
      - Removes dead agents

      \returns A string, ready to be written to log file
     */
    void Population::step(const size_t timeStep, std::vector<float> &result_phen,std::vector<float> &result_gen,bool &failed)
    {

        size_t num_food = 0;
        size_t num_food_bt[FOOD_TYPES];
        for(int i=0;i<FOOD_TYPES;i++)num_food_bt[i]=0;

        std::ostringstream Agent_str_stats; // stream for ofstream
        std::ostringstream Agent_str_stats_gen; // stream for ofstream
        std::ostringstream Env_str_stats; // stream for ofstream
        std::ostringstream Reprod_str_stats; // stream for ofstream
        std::ostringstream Forage_str_stats; // stream for ofstream

        bool lock_food(false);
#ifdef FOOD_LOCK_SKILL
        //lock_food=(timeStep%LOCK_INTERVAL==0);
        lock_food=false; // food should remain unlocked, until it is consumed.
#endif

        count_and_lock_food_fields(num_food,num_food_bt,timeStep,lock_food);

        if(timeStep%100==0)
            std::cout<<"## "<<popID_<<" timestep "<<timeStep<<" population is "<<population_.size()<<" food "<<num_food<<" numfood0 "<<num_food_bt[0]<<" numfood1 "<<num_food_bt[1]<<std::endl;

        log_env_stats(Env_str_stats,timeStep);
#ifdef VISUALIZE
//    if(((int)timeStep%100)==0){
        output_food_fields(timeStep);
        output_agents(timeStep);
//    }
#endif

#ifdef DEBUG
        assert(num_food==total_num_food_cells_);
#endif
#ifndef IMMORTALS
        remove_dead_agents();
        reproduction(timeStep,Reprod_str_stats,life_bonus_); // replicate agents
#endif
        if(population_.size()> nmax_){
            cap_population();  // remove extra agents
        }else if(population_.size()==0){
            failed=true;
            return;
        }
        std::vector<population_type*> ppopulation;
        shuffle_agents_pointers(ppopulation);
        //Ugly workaround
        // TODO can this be merged with the following loop?
        #pragma omp parallel for default(none) shared(ppopulation) schedule(dynamic,1)
        for (size_t i=0;i<ppopulation.size();i++) {
            Agent& agent=get_agent_at(*ppopulation[i]);
            const int fieldID = get_field_at(*ppopulation[i]);
            const Genome::perception_type perceptions = compute_agent_perceptions(*ppopulation[i],fieldID);
            int action = agent.choose_action(perceptions,binary_in_,timeStep);
            agent.set_action(action);
            agent.birthday();
        }

        for (auto &acontain : ppopulation){
            Agent& agent=get_agent_at(*acontain);
            int action=agent.get_action();

                if(action==Agent::aton("north") ||
                   action==Agent::aton("west") ||
                   action==Agent::aton("east") ||
                   action==Agent::aton("south")) {
                    // just move
#ifdef DEBUG
                    std::cout<<"Agent "<<get_agent_at(acontain).get_ID()<<" MOVES "<<Agent::ntoa(action)<<std::endl;
#endif //DEBUG
                    agent.give_reward(-MOVE_ENERGY);
                    agent_moves(*acontain,action);
                } else if(action==Agent::aton("eat")) {
                    // eat
#ifdef DEBUG
                    std::cout<<"Agent "<<get_agent_at(acontain).get_ID()<<" EATS "<<Agent::ntoa(action)<<std::endl;
#endif //DEBUG
                    agent.vary_energy(-EAT_ENERGY);
                    agent_eats(*acontain,num_food,num_food_bt,timeStep,Forage_str_stats);
                } else if(action==-1) {
#ifdef DEBUG
                    int fieldID = get_field_at(*acontain);
                    std::cout<<"Agent "<<get_agent_at(*acontain).get_ID()<<" in field "<<fieldID<<" is dead, code: "<<action<<std::endl;
#endif //DEBUG
                }else{
                    std::cout<<"Warning: unknown action"<<std::endl;
                }
            }
        replenish_food(timeStep,num_food, num_food_bt);

        result_phen.reserve(population_.size());
        result_gen.reserve(population_.size());
        for (auto &acontain : population_){
            Agent& agent=get_agent_at(acontain);
            result_phen.push_back(agent.get_skill(0));
            result_gen.push_back(agent.get_skill_gen(0));
        }

        //Logging loop should be parallelized
        if(timeStep%LOG_FREQ==0){
            //Ugly workaround //This improves performance ONLY with large number of agents per thread (agents >> threads)
            for (auto &acontain : population_){
                Agent& agent=get_agent_at(acontain);
#ifdef LEARN
                Genome::actions_type result = agent.test_agent(true);
                log_agent_stats(agent,get_field_at(acontain),result,timeStep,Agent_str_stats);
#endif
                Genome::actions_type result_gen = agent.test_agent(false);
                log_agent_stats(agent,get_field_at(acontain),result_gen,timeStep,Agent_str_stats_gen);
            }

            log_WRITE(Agent_str_stats,statsFileAgent);
            log_WRITE(Agent_str_stats_gen,statsFileAgent_gen);
            log_WRITE(Env_str_stats,statsFileEnv);
            log_WRITE(Reprod_str_stats,statsFileReprod);
        }
        log_WRITE(Forage_str_stats,statsFileForage);
    }

    //! Move the agent of one cell in direction 'orientation'.
    /*!

      \param [out] a The object containing the agent and its position. Will be updated by the function
      \param [in] orientation The numeric value indicating in which direction the agent wants to move
      If direct_feedback is enabled, the agent receives a small reward if it enters a cell with food. It could be useful while learning.
     */
    void Population::agent_moves(population_type &a,int orientation) //{NORTH=0,WEST=1,EAST=2,SOUTH=3}
    {
        int fieldID = get_field_at(a);
        std::vector<int> neighbors = get_neighboring_cells(1, fieldID, field_size_, field_size_ * field_size_); // indexes of neighbors
        std::vector<int> idxs = {1, 3, 5, 7};
        int dest_idx = neighbors[idxs[orientation]]; // moves in the direction defined by orientation
        // update field reference in population
        a.second=dest_idx;
        // decrease origin counter
        assert(fields_[fieldID].get_num_agents()>0);
        fields_[fieldID].rem_agent();
        // increase destination counter
        fields_[dest_idx].add_agent();
        // if direct_feedback enabled then add ENTER_ENERGY to the agent when he enters the cell!
        if(direct_feedback_){
          //FOOD - direct feedback (CHECK)
          if(fields_[fieldID].get_all_food()>0){
            get_agent_at(a).give_reward(ENTER_ENERGY);
          }
        }
#ifdef DEBUG
        std::string ori = Agent::ntoa(orientation);
        std::cout<<"Agent "<<get_agent_at(a).get_ID()<<" moves "<<ori<<" from field "<<fieldID<<" to field "<<dest_idx<<" and neighbors are "<<neighbors[idxs[0]]<<", "<<neighbors[idxs[1]]<<", "<<neighbors[idxs[2]]<<", "<<neighbors[idxs[3]]<<std::endl;
#endif //DEBUG
    }

    //! Eat from the current cell.
    /*!
      \param [out] a The object containing the agent and its position. Will be updated by the function
      \param [in] num_food The number of cells with food in the environment. Spawn new food sources if this number is smaller than what desired

      The agent might want to eat even if there is no food.
      Update appropriatedly the energy of the agent and respawn a new food source if the current one is empty.
      If XXXXCHANGE is set, there is a probability of failing the foraging (prey escapes) that gets reduced with an increasing number of agents.
      If foraging fails, the agent gets a penalty of food_kill_energy_

      Side effect: The function updates the cells in the environment vector
     */
    void Population::agent_eats(population_type &a,size_t &num_food,size_t (&by_type_cells_w_food)[FOOD_TYPES], const size_t timeStep,  std::ostringstream &str_stats)
    {
        //std::cout<<"######Agent "<<get_agent_at(a).get_ID()<<" tries to eat with energy "<<get_agent_at(a).get_energy()<<" eats "<<" from field "<<get_field_at(a)<<" with food "<<fields_[get_field_at(a)].get_food(0)<<" , "<<fields_[get_field_at(a)].get_food(1)<<" and agents "<<fields_[get_field_at(a)].get_num_agents()<<std::endl;
        //FOOD - (CHECK) POLICY
        bool is_food(false);
        std::vector<size_t> food_t(FOOD_TYPES);
        for (size_t idx=0;idx<FOOD_TYPES;idx++){
            food_t[idx]=idx;
        }
        std::random_shuffle(food_t.begin(),food_t.end());

        for(auto idx:food_t){
            is_food = fields_[get_field_at(a)].get_food(idx)>0;
            if(is_food) {
#ifdef SKILL_TO_EAT
                bool use_skill=true;
#else
                bool use_skill=false;
#endif
                bool success=agent_tries_to_eat(a,num_food,by_type_cells_w_food,idx,use_skill);
                log_forage_stats(get_agent_at(a),timeStep,success,idx,str_stats);
                if(success)
                    break;
            }
        }
#ifdef DEBUG
        if(!is_food){
            std::cout<<"Current cell has no food";
        }
        std::cout<<", new energy level is "<<get_agent_at(a).get_energy()<<std::endl;
#endif //DEBUG
    }

    bool Population::agent_tries_to_eat(population_type &a,size_t &num_food,size_t (&by_type_cells_w_food)[FOOD_TYPES],size_t food_type,bool use_skill){
        //FOOD (CHECK) THRESHOLD PARAMETERS
        bool eat_successful(false);

        eat_successful=!use_skill || is_agent_successful(a,food_type);

#ifdef DEBUG
        std::cout<<"Agent "<<get_agent_at(a).get_ID() <<" with energy "<<get_agent_at(a).get_energy()<<" eats "<<(eat_successful ? "successfully" : "unsuccessfully")<< " food of type "<< food_type<<" from field "<<get_field_at(a)<<" with food "<<fields_[get_field_at(a)].get_food(0)<<" , "<<fields_[get_field_at(a)].get_food(1)<<" and agents "<<fields_[get_field_at(a)].get_num_agents()<<std::endl;
#endif //DEBUG
#ifdef FOOD_LOCK_SKILL
        if(fields_[get_field_at(a)].is_food_unlocked(food_type)) { // the food was already unlocked
            eat_successful=true; // eat without effort
            double thresh_skillful(0.35);
            if(get_agent_at(a).get_skill(food_type)<thresh_skillful) // it wouldn't be likely for this agent to eat this food if it were locked
                fields_[get_field_at(a)].inc_times_shared(food_type); // consider the food as shared
        } else if(eat_successful){  // is able to forage, thus to unlock
            fields_[get_field_at(a)].unlock_food(food_type); // unlock the food source for the next agents
            fields_[get_field_at(a)].inc_times_unlocked(food_type); // increment the counter
            // TODO why having a threshold?
        } else {
            // bad luck
        }
#endif  // FOOD_LOCK_SKILL
         if(eat_successful){
                get_agent_at(a).give_reward(food_energy_); // gain energy
                fields_[get_field_at(a)].consume_bundle(food_type); // consume food
#ifdef LEARN
#ifdef DEBUG
                std::cout<<"Changing skill of agent "<<get_agent_at(a).get_ID()<<" from "<<get_agent_at(a).get_skill(0);
#endif
                get_agent_at(a).increase_skill(food_type,SKILL_INCREASE); // increase skill by eating
#ifdef DEBUG
                std::cout<<" to "<<get_agent_at(a).get_skill(0)<<std::endl;
#endif
#endif
                if(fields_[get_field_at(a)].get_food(food_type)<=0){ // if the food resource is empty spawn a new one
                    fields_[get_field_at(a)].set_initial_food(food_type,0); // reset the initial food variable
                    assert(num_food>0);
                    assert(by_type_cells_w_food[food_type]>0);
                    num_food--;
                    by_type_cells_w_food[food_type]--;
                    }
#ifdef DEBUG
                std::cout<<"Eat successful, remaining food "<<fields_[get_field_at(a)].get_food(0)<<" , "<<fields_[get_field_at(a)].get_food(1)<<std::endl;
#endif //DEBUG
                return true;
            }else{
                //FOOD (CHECK) SHOULD BE TREATED WITH EAT ENERGY ALREADY (BETTER)
                //get_agent_at(a).give_reward(-food_energy);
#ifdef DEBUG
                std::cout<<"Feeding unsuccessful, getting damage"<<std::endl;
#endif //DEBUG
                return false;
            }
        return false;
    }

    bool Population::is_agent_successful(const population_type &a,const size_t &food_type)const{
        //FOOD (CHECK) THRESHOLD PARAMETERS
        std::uniform_real_distribution<double> dist_draw(0,1);

#ifdef COMPETITION
        double success_threshold(get_agent_at(a).get_skill(food_type)/(double)std::max(std::log(fields_[get_field_at(a)].get_num_agents()),1.0));
#elif defined COOPERATION
        double success_threshold(1.-((1.-get_agent_at(a).get_skill(food_type))/(double)std::max(std::log(fields_[get_field_at(a)].get_num_agents()),1.0)));
#else
        double success_threshold(get_agent_at(a).get_skill(food_type));
#endif
#ifdef NONLINEAR_PROB
        double prob=success_threshold*success_threshold;
#else
        double prob=success_threshold;
#endif
        return dist_draw(rng)<=prob;
    }

    Agent Population::agent_replicates(Agent &a)
    {
#ifdef DEBUG
            std::cout<<"Agent "<<a.get_ID()<<" with energy "<<a.get_energy()<<" replicates in field "<<f;
#endif //DEBUG
            Agent offspring = a.replicate(agent_counter_++);
            offspring.set_energy(a.get_energy()/2.0);
            a.set_energy(a.get_energy()/2.0);             // half energy of self
#ifdef DEBUG
            std::cout<<" to offspring "<<offspring.get_ID()<<", new energy of agent "<<a.get_energy()<<" and offspring "<<get_agent_at(population_.end()-1).get_energy()<<std::endl;
        } else {
            std::cout<<"Agent "<<a.get_ID()<<" with energy "<<a.get_energy()<<" has not enough energy to replicate"<<std::endl;
#endif //DEBUG
        return offspring;
    }

    //! Remove one agent
    /*!
      \param [in] it Pointer to the agent to be deleted

      Side effect: The function updates the population vector
     */
    void Population::del_agent(std::vector<population_type>::iterator &it)
    {
#ifdef DEBUG
        std::cout<<"deleting agent "<<get_agent_at(*it).get_ID()<<std::endl;
#endif //DEBUG
        std::swap(*it , *(population_.end()-1)); // swap the element with the last element
        population_.pop_back();
        it--;                   // go back one step as the element at the iterator has not been processed yet
    }

    void Population::del_agent(int i)
    {
#ifdef DEBUG
        std::cout<<"deleting agent "<<get_agent_at(i).get_ID()<<std::endl;
#endif //DEBUG
        std::vector<population_type>::iterator it;
        it=population_.begin()+i;
        std::swap(*it , *(population_.end()-1)); // swap the element with the last element
        population_.pop_back();
        it--;                   // go back one step as the element at the iterator has not been processed yet
    }

    //! randomly shuffle the population
    void Population::shuffle_agents()
    {
        std::random_shuffle(population_.begin(),population_.end());
    }

    //! randomly shuffle the population
    void Population::shuffle_agents_pointers(std::vector<population_type*> &ppopulation)
    {
        ppopulation.clear();
        ppopulation.reserve(population_.size());
        for(auto &elem:population_){
            ppopulation.push_back(&elem);
        }
        std::random_shuffle(ppopulation.begin(),ppopulation.end());
    }

    //! List of cells in the neighborhood.
    /*!
      \returns A vector of indexes of cells in the Moore neighborhood of the cell at index, including the cell at index (I)
      The return vector maps to the environment in the following way:
      (0,1,2
       3,I,5
       6,7,8)
     */
    std::vector<int> Population::get_neighboring_cells(const int radius, /*!< How far can the agent see */
                                                       const int index, /*!< The index of the current cell */
                                                       const int grid_size, /*!< size of the environment */
                                                       const int num_cells /*!< number of cells in the environment */
                                                       )const {
        std::vector<int> neighbors;
        int val(0);
        for (int y = -radius; y <= radius; y++) {
            for (int x = -radius; x <= +radius; x++) {
                int round = (int)(index/grid_size*grid_size); // first element in the row
                val=((index+x+grid_size)%grid_size+ // periodic boundaries in the row
                                    (round+y*grid_size+num_cells)%num_cells); // periodic boundaries in the column
                assert(val>=0);
                neighbors.push_back(val);
            }
        }
        return neighbors;
    }

    //! Returns the field of view from the current cell
    /*!
      takes the references to the cells around the agent in a range fov_radius and groups them in 5 vectors that represent the cardinal directions + current cell
      | F | F | F | F | F |
      | L | F | F | F | R |
      | L | L | H | R | R |
      | L | B | B | B | R |
      | B | B | B | B | B |

      \returns A bidimensional vector that contains:
      [cells-ahead, cells-left, cells-here, cells-right, cells-back]
    */
    std::vector<std::vector<int>> Population::field_of_view(const int fov_radius, /*!< How far can the agent see */
                                                            const int index, /*!< The index of the current cell */
                                                            const int grid_size, /*!< size of the environment */
                                                            const int num_cells /*!< number of cells in the environment */
                                                            ) const
    {
        std::vector<int> neighs = get_neighboring_cells(fov_radius,index,grid_size,num_cells);
        int matrix_dim=fov_radius*2+1;
        std::vector<std::vector<int>> result(5);
        for(int y=0;y<matrix_dim;y++)
            for(int x=0;x<matrix_dim;x++)
                {
                    if(x==y && x==(matrix_dim-1-y)){ // central cell
                        result[2].push_back(neighs[y*matrix_dim+x]);
                    } else if (x>=y && x<=(matrix_dim-1-y)) { // north
                        result[0].push_back(neighs[y*matrix_dim+x]);
                    } else if (x<=y && x>=(matrix_dim-1-y)) { // south
                        result[4].push_back(neighs[y*matrix_dim+x]);
                    } else if (x>y && x>(matrix_dim-1-y)) { // east
                        result[3].push_back(neighs[y*matrix_dim+x]);
                    } else {    // west
                        result[1].push_back(neighs[y*matrix_dim+x]);
                    }
                }
        return result;
    }

    int Population::count_food(const Field field,const bool (&see_successful)[FOOD_TYPES])const{
        /**
         * Returns the quantity of visible food
         * If skill is used to see, it determines the quantity of food to be seen
         * Is skill is not used to see, it returns all food
         */
        int counter=0;
//FOOD (Check) policy (currently they can see any food without knowing the type)
#ifdef SKILL_TO_SEE             // can see only some food
        for(int idx=0;idx<FOOD_TYPES;idx++){
            counter+=Population::count_food_TYPE(idx,field,see_successful);
        }
// #elif defined FOOD_LOCK_SKILL
//         counter=field.get_all_unlocked_food(); // can see only the unlocked food
#else //SKILL_TO_SEE
        counter=field.get_all_food(); // can see all the food
#endif //SKILL_TO_SEE
        return counter;
    }

    int Population::count_food_TYPE(const int food_type,const Field field,const bool (&see_successful)[FOOD_TYPES]) const{
        /**
         * Returns the quantity of visible food, for a given type
         * If skill is used to see, it determines the quantity of food to be seen
         * Is skill is not used to see, it returns all food
         * In this last case, whether locking is enable or disabled should not make a difference: The agent will see all food, try to eat the locked food and fail. If instead the agent would only see the unlocked food, it will never fail the foraging.
         */
        int counter=0;
        assert(food_type<FOOD_TYPES);
#ifdef FOOD_LOCK_SKILL
        bool unlocked=field.is_food_unlocked(food_type); // if unlocked it is seen with p=1
#else //FOOD_LOCK_SKILL
        bool unlocked=false; // evaluate the second part of the logical condition
#endif //FOOD_LOCK_SKILL
        if(unlocked||see_successful[food_type]){
            counter+=field.get_food(food_type);
        }
        return counter;
    }

    int Population::count_agents(const Field field,const bool same_field) const{
        int counter=0;
#ifdef FILTER_STATIC_PERCEPTIONS
        //FOOD (Check) Consider agents eating disregarding the food type
        //This assumes that the other agents have already learned to stop and eat whenever there is food (not good)
        int food=0;
#ifdef FOOD_LOCK_SKILL
        food=field.get_all_unlocked_food();
#else
        food=field.get_all_food();
#endif
        if(food>0){ // Only if agents are eating
            counter=field.get_num_agents();
            if(same_field) counter-=1;
            // TODO What if there are two resources in one cell? which one will the agent consume?
        }
#else // FILTER_STATIC_PERCEPTIONS
        // Consider all agents
        counter=field.get_num_agents();
        if(same_field) counter-=1;
#endif // FILTER_STATIC_PERCEPTIONS
        return counter;
    }

    //! Returns the perception vector of an agent in the given field
    /*!
      \returns A Genome object (vector<double>) that contains for each of the visible 5 neighbors (including the current field) the amount of food and the number of agents
      food-north, food-west, food-here, food-east, food-south
      if INTERACT is defined the vector also contains:
      agents-north, agents-west, agents-here, agents-east, agents-south
      if INVISIBLE_FOOD is defined the vector looks like:
      agents-north, agents-west, FOOD-here, agents-east, agents-south
    */
    Genome::perception_type Population::compute_agent_perceptions(const population_type &a,const int fieldID) const // TODO convert the result type into a hashmap?
    {
        Genome::perception_type perceptions; // ignore input1
        std::vector<std::vector<int> > neighbors = field_of_view(fov_radius_,fieldID,field_size_,field_size_*field_size_); // indexes of neighbors
        //TODO if the grid is very small, because of boundary conditions agents can be counted twice... it should never be a problem
        int counter=0;


        bool see_successful[FOOD_TYPES];
        for(size_t idx=0;idx<FOOD_TYPES;idx++){
#ifdef SKILL_TO_SEE
            see_successful[idx]=is_agent_successful(a,idx);
#else
            see_successful[idx]=true;
#endif
        }

        // Generate a vector that contains the following information:
        // Food_{N,W,H,E,S}, agents_{N,W,H,E,S}
        // The order is determined by what returned by field_of_view()
        std::vector<std::string> labels = {"foodN", "foodW","foodH", "foodE", "foodS", "agentN", "agentW", "agentH", "agentE", "agentS","foodH0","foodH1"};
        // fill in inputs 0 to 4: food
        for(auto &v:neighbors) {
            counter=0;
            for(auto &f:v){
                counter+=count_food(fields_[f],see_successful);
            }
            perceptions.push_back(counter);
        }
        // fill in inputs 5 to 9: agents
        for(auto &v:neighbors) {
            counter=0;
            // Consider only static agents (eating)
            for(auto &f:v) {
                counter+=count_agents(fields_[f],fieldID==f);
            }
            perceptions.push_back(counter);
        }

        //Populate foodH0 and foodH1
        std::vector<int> dv=neighbors[2];
        counter=0;
        for(auto &f:dv) {
            counter+=count_food_TYPE(0,fields_[f],see_successful);
        }
        perceptions.push_back(counter);

        dv=neighbors[2];
        counter=0;
        for(auto &f:dv) {
            counter+=count_food_TYPE(1,fields_[f],see_successful);
        }
        perceptions.push_back(counter);

        //perceptions[7]-=1;      // do not count the agent in the current cell

        // Get the perception types
        std::vector<std::string> names=visualsys.names;
        // generate a new vector with only the perceptions that are allowed
        Genome::perception_type perceived; // what the agent actually sees
        for(auto &i : names){
            int pos = find(labels.begin(), labels.end(), i) - labels.begin(); // find the corresponding value in the first vector
            assert(pos>=0);
            assert((size_t) pos < labels.size());
            perceived.push_back(perceptions[pos]);
        }
        return perceived;
    }

    //Old version, it gives the same results
    Genome::perception_type Population::compute_agent_perceptions2(const population_type &a,const int fieldID) const // TODO convert the result type into a hashmap?
    {
        Genome::perception_type perceptions; // ignore input1
        std::vector<std::vector<int> > neighbors = field_of_view(fov_radius_,fieldID,field_size_,field_size_*field_size_); // indexes of neighbors
        //TODO if the grid is very small, because of boundary conditions agents can be counted twice... it should never be a problem
        int counter=0;


        bool see_successful[FOOD_TYPES];
        for(size_t idx=0;idx<FOOD_TYPES;idx++){
#ifdef SKILL_TO_SEE
            see_successful[idx]=is_agent_successful(a,idx);
#else
            see_successful[idx]=true;
#endif
        }


        // TODO generate from struct constants_t and change the order of inputs to match that of the actions in the struct. (update also tests to match this new order)
#ifndef INVISIBLE_FOOD
        // fill in inputs 0 to 4: food
        for(auto &v:neighbors) {
            counter=0;
            for(auto &f:v){
                counter+=count_food(fields_[f],see_successful);
            }
            perceptions.push_back(counter);
        }
#endif //INVISIBLE_FOOD
#ifdef INTERACT
        // fill in inputs 5 to 9: agents
        for(auto &v:neighbors) {
            counter=0;
    // Consider only static agents (eating)
            for(auto &f:v) {
                counter+=count_agents(fields_[f],fieldID==f);
            }
            perceptions.push_back(counter);
        }
#ifdef INVISIBLE_FOOD
        // substitute information about agents in the current cell with info about food
        counter=0;
        for(auto &f:neighbors[2]){
            //FOOD (Check) currently they can get information of the total food
            counter+=count_food(fields_[f],see_successful);
        }
        perceptions[2]=counter;
#elif !defined INVISIBLE_FOOD
//        perceptions[7]-=1;      // do not count the agent in the current cell
//CHECK        assert(perceptions[7]>=0);
#endif //INVISIBLE_FOOD
#endif //INTERACT
        //// ---- some debug code ----
#ifdef DEBUG
#ifdef INVISIBLE_FOOD
#ifdef INTERACT
        std::cout<<"Perceptions: agent_north="<<perceptions[0]<<" agent_west="<<perceptions[1]<<" food_here="<<perceptions[2]<<" agent_east="<<perceptions[3]<<" agent_south="<<perceptions[4]<<std::endl;
#endif // INTERACT
#elif !defined INVISIBLE_FOOD
#ifdef INTERACT
        std::cout<<"Perceptions: food_north="<<perceptions[0]<<" food_west="<<perceptions[1]<<" food_here="<<perceptions[2]<<" food_east="<<perceptions[3]<<" food_south="<<perceptions[4]<<"agent_north="<<perceptions[5]<<" agent_west="<<perceptions[6]<<" agent_here="<<perceptions[7]<<" agent_east="<<perceptions[8]<<" agent_south="<<perceptions[9]<<std::endl;
#elif !defined INTERACT
        std::cout<<"Perceptions: food_north="<<perceptions[0]<<" food_west="<<perceptions[1]<<" food_here="<<perceptions[2]<<" food_east="<<perceptions[3]<<" food_south="<<perceptions[4]<<std::endl;
#endif // INTERACT
#endif //INVISIBLE_FOOD
#endif  // debug
        return perceptions;
    }

    //! Initialize the grid
    void Population::initialize_fields() {
        for (size_t i = 0; i < field_size_*field_size_; i++) {
            fields_.push_back(Field());
        }

#ifdef DEBUG
        std::cout<<"Debugging fields..."<<std::endl;
        for(auto &f : fields_) {
            assert(&f!=NULL);    // object has been created
            assert(f.get_num_agents()==0); // contains 0 agents
            //XXXXFOOD
            for(size_t idx=0;idx<FOOD_TYPES;idx++){
                assert(f.get_food(idx)==0);       // contains 0 food
            }
        }
#endif //DEBUGGING
    }

    void Population::count_and_lock_food_fields(size_t &cells_w_food,size_t (&by_type_cells_w_food)[FOOD_TYPES],const size_t &timeStep, bool lock){
        // increase food in fields
        for (size_t i = 0; i < fields_.size(); i++){
            //FOOD (CHECK)
            for(size_t idx=0;idx<FOOD_TYPES;idx++){
                int food = fields_[i].get_food(idx);
                if (food > 0){
                    by_type_cells_w_food[idx]++;
                    cells_w_food++;
                }
                // ---------- log information about fields ----------
                if (food > 0) {
                    //Lock any unlocked food source
                    if(lock)
                        fields_[i].lock_food(idx);
                }
            }
        }
    }

    //! Initialize the population randomly
    void Population::initialize_population(std::string load_pop,std::string _enotype,float mutation_size_skill_,float mutation_size_weigths_) {
        if(load_pop.empty()){
        population_.reserve(n0_);
        std::uniform_int_distribution<int> fields_index_distribution(0,field_size_*field_size_-1);
        // TODO put into function
        for (size_t i = 0; i < n0_; i++) {
            Agent ag = Agent(agent_counter_++,mutation_size_skill_,mutation_size_weigths_,prob_eat_success_);
            size_t idx = fields_index_distribution(rng);    // place in random cell
            population_.push_back(population_type(ag,idx)); // add to list
            assert(idx<fields_.size());
            fields_[idx].add_agent();
        }
#ifdef DEBUG
        for_each(population_.begin(),population_.end(),[this](population_type a){std::cout<<"Agent "<<get_agent_at(a).get_ID()<<" in field "<<get_field_at(a)<<std::endl;});
        std::cout<<"debugging population...";
        assert((size_t) std::accumulate(fields_.begin(),
                               fields_.end(),
                               0,
                               [](int &a, Field &b){return a+=b.get_num_agents();}) == n0_); // num of agents if fields is consistent with number of agents
        std::vector<int>counts(field_size_*field_size_,0);
        for(auto & a : population_) {
            counts[get_field_at(a)]++;
        }
#endif //DEBUG
        } else {
            population_.reserve(nmax_);
            std::ostringstream filename;
            filename << load_pop<<"_"<<popID_<<".xml";
            std::cout << "Initializing population from "<<filename.str()<<"\n";
            load_population(filename.str());
            std::cout<<"Rank "<<popID_<<" initialized "<<population_.size()<<" agents"<<std::endl;
            if(population_.size()>0) {
            for(auto &a: population_){
                // reset ids sequentially
                get_agent_at(a).set_ID(agent_counter_++);
                // add them to the fields
                fields_[get_field_at(a)].add_agent();
                // initialize the logic
                if(_enotype=="genotype"){
                    get_agent_at(a).use_genotype();
                }else if(_enotype=="phenotype"){
                    get_agent_at(a).use_phenotype();
                }else if(_enotype.empty()){
                    //do nothing
                }else {
                    std::cout << "Warning, option "<<_enotype<<" not understood" << "\n";
                }
            }
            } else {
                std::cout << "Rank "<< popID_<<" skips empty population" << "\n";
    }
        }
    }

    void Population::reproduction(size_t timeStep,std::ostringstream &Reprod_str_stats,double life_bonus) {
        /**
           Choose agents that reproduce at this timestep using the roulette wheel selection mechanism with stochastic acceptance:
           The probability of reproducing is the ratio between the individual fitness and ENERGY_MAX
         */
        // Get the fitness of all individuals
        int max_energy_=max_age_*(1-life_bonus); // how much reproduction if favored over death
        if(population_.size()==0) { // generate a new population
#ifdef RESPAWN
            std::cout<<"Warning: all agents are dead, spawning a new population"<<std::endl;
            initialize_population("","");
            seed_population();
#endif
            // exit(666);
        }else {
            std::uniform_real_distribution<double> dist01(0,1);
            std::vector<population_type> offsprings;
            std::vector<population_type> parents;
            // Separate population and offsprings to avoid trouble with iterators: the iterators get invalidated when the population is resized
            for(auto &a : population_){
                float weight=((float)get_agent_at(a).get_energy()/max_energy_);
                if(dist01(rng)<weight){ // accept with probability proportional to age and inversely proportional to fitness
#ifdef DEBUG
                    std::cout<<"Agent "<<get_agent_at(a).get_ID()<<" with energy "<<get_agent_at(a).get_energy()<<" reproduces"<<std::endl;
#endif
                    if(get_agent_at(a).get_energy()>1){
                        Agent offspring=agent_replicates(get_agent_at(a));
                        if(offspring.get_ID()!=-9){
                            offsprings.push_back(population_type(offspring,get_field_at(a))); // add agent to list
                            parents.push_back(a); // add agent to list
                        }else{
                            std::cout<<"Warning: invalid offspring"<<std::endl;
                        }
                    }
                }
                    }
            for(uint i=0;i<offsprings.size();i++){ // merge with population, loop is on offspring vector which does not get resized
                population_.push_back(offsprings[i]);
                fields_[get_field_at(offsprings[i])].add_agent(); // increase the a counter
                log_reprod_stats(get_agent_at(offsprings[i]),get_agent_at(parents[i]),timeStep,Reprod_str_stats);
            }
        }
    }

    void Population::replenish_food(int tStep,size_t &num_food,size_t (&num_food_bt)[FOOD_TYPES]){

        //FOOD
        //Define amount of each food source
        int day_of_season(tStep%season_length_);
        int season_num(tStep/season_length_);
        //int amount_needed(0);
        set_food_quantities(day_of_season,season_num);
        int to_add[FOOD_TYPES];
        assert(FOOD_TYPES>0);
        // check how many food sources there are
        bool lock_food(false);
#ifdef FOOD_LOCK_SKILL
        lock_food=true;
#endif
        if(num_food<total_num_food_cells_){ // need to add some food
            for(size_t idx=0;idx<FOOD_TYPES;idx++){ // every food type
                to_add[idx]=0;
                if(num_food_bt[idx]<num_food_cells[idx]){ // if there isn't enough of that type
                    to_add[idx]=std::min(total_num_food_cells_-num_food, // total quantity of food to add
                                         num_food_cells[idx]-num_food_bt[idx]); // difference between wished and current quantity of this type of food
                    for( size_t i = 0; i < (size_t) to_add[idx]; ++i ) // for how many missing food sources
                        create_food_source(fields_,1,food_max_,0,idx,FOOD_EXCLUSIVE,lock_food); // add food of that type
                }
            }
        }
#ifdef DEBUG
        size_t food_sum=0;
        size_t foot_bt[FOOD_TYPES];
        for(size_t idx=0;idx<FOOD_TYPES;idx++) foot_bt[idx]=0;
        for(auto &f : fields_) {
            for(size_t idx=0;idx<FOOD_TYPES;idx++){
                if(f.get_food(idx)!=0){
                    foot_bt[idx]++;
                    food_sum++;
                }
            }
        }
        std::cout<<"#AMOUNT amount of food0 "<<foot_bt[0]<<" food1 "<<foot_bt[1]<<" final "<<food_sum<<" TOTAL "<<total_num_food_cells_<<std::endl;
        assert(food_sum==total_num_food_cells_);
        std::cout<<"Correct AMOUNT amount of food!"<<std::endl;
#endif //DEBUG

    }

void Population::set_food_quantities(int day_of_season,int season_num){

        assert(FOOD_TYPES==2);//change as the number of food cells change
        assert(food_proportion_<=1);
        size_t food_start(0);
#ifdef SEESAW_SEASON
        food_start=size_t((double) total_num_food_cells_*food_proportion_);
#else
        if(season_num%2==0){
            food_start=size_t((double) total_num_food_cells_*food_proportion_);
        }else{
            food_start=size_t((double) total_num_food_cells_*(1.0-food_proportion_));
        }
#endif
        num_food_cells[0]=food_start;
        num_food_cells[1]=total_num_food_cells_- num_food_cells[0];
#ifdef DEBUG
        std::cout<<"AMOUNT amount of food0 "<<num_food_cells[0]<<" amount of food1 "<<num_food_cells[1]<<std::endl;
#endif //DEBUG
    }

void Population::initial_spawn_food(){
        //FOOD
        //Define amount of each food source
        assert(FOOD_TYPES==2);//change as the number of food cells change
        assert(food_proportion_<=1);
        num_food_cells[0]=size_t((double) total_num_food_cells_ * food_proportion_);
        num_food_cells[1]=total_num_food_cells_- num_food_cells[0];

        assert(FOOD_TYPES>0);
        for(size_t idx=0;idx<FOOD_TYPES;idx++){
            // spawn initial food
            for( size_t i = 0; i < num_food_cells[idx]; ++i ){
                create_food_source(fields_,1,food_max_,0,idx,FOOD_EXCLUSIVE,false); // no init food, pick type at random
            }
        }
#ifdef DEBUG
        size_t food_sum=0;
        for(auto &f : fields_) {
            for(size_t idx=0;idx<FOOD_TYPES;idx++){
                if(f.get_food(idx)!=0)
                    food_sum++;
            }
        }
        assert(food_sum==total_num_food_cells_);
        std::cout<<"Correct INITIAL amount of food!"<<std::endl;
#endif //DEBUG
    }

    //!  Death predicate
    /*!
      This class contains the function that decides whether to delete an agent
    */

    class DeathPredicate
    {
    public:
        DeathPredicate( double probability ) : probability_(probability) {}
        bool operator()( Agent const& a ) const
        {
            std::uniform_real_distribution<double> distribution_01(0,1);
            return probability_ > 1. || a.is_dead();// || distribution_01(rng) < probability_;
        }

    private:
        double probability_;
    };

    void Population::remove_dead_agents() {
        /**
           Remove agents using the roulette wheel selection algorithm via stochastic acceptance.
           Agents are drawn uniformly at random. Agents with higher age and lower energy are more likely to be killed.
           The number of agents removed is stochastic and between 1 and population.size*FRACTION_DEATHS
         */
        //remove dead agent
        population_.erase(std::remove_if(population_.begin(), population_.end(),
                                         [this](Population::population_type &a) {
                                             bool dead=get_agent_at(a).is_dead();
                                             if(dead){
                                                 //std::cout<<"removing zombie "<<get_agent_at(a).get_ID()<<" with energy "<<get_agent_at(a).get_energy()<<" and age "<<get_agent_at(a).get_age()<<std::endl;
                                                 fields_[get_field_at(a)].rem_agent(); // remove the agent from the field
                                             }
                                             return dead;
                                         }), population_.end());
        size_t nkills=(int)(FRACTION_DEATHS*population_.size());
        if(nkills==0)
            nkills=population_.size(); // test all the population
#ifdef DEBUG
        if(population_.size()!=0){
            std::cout<<"Population has size "<<population_.size()<<". Removing "<<nkills<<" agents from the population"<<std::endl;
        }
#endif
        std::uniform_real_distribution<double> dist01(0.0,1.0);
        // reshuffle a vector of pointers to avoid the cost of moving objects around
        std::vector<population_type*> ppopulation;
        shuffle_agents_pointers(ppopulation);
        // test the first random 'nkills' agents and remove them from the game
        // the agents are actually not removed from the vector because, by removing one agent, all pointers will be invalidated. They are saved to the vector 'tokill'.
        std::vector<int> tokill;
        for (size_t i=0;i<nkills;i++) {
            float weight=((float)get_agent_at(*ppopulation[i]).get_age()/max_age_);
            if(dist01(rng)<weight){ // accept with probability proportional to age and inversely proportional to fitness
#ifdef DEBUG
                std::cout<<"Agent "<<get_agent_at(*ppopulation[i]).get_ID()<<" with energy "<<get_agent_at(*ppopulation[i]).get_energy()<<" and age "<<get_agent_at(*ppopulation[i]).get_age()<<" is removed from the population"<<std::endl;
#endif
                fields_[get_field_at(*ppopulation[i])].rem_agent(); // remove the agent from the field
                get_agent_at(*ppopulation[i]).set_energy(-1);
                auto idx=std::distance(&population_[0],ppopulation[i]);
                tokill.push_back(idx);
        }
            }
        // remove the agents from the population vector. When an element is removed, it is swapped with the last and the vector is popped, thus all pointers after it are invalidated.
        // Start from the last position so that pointers keep being valid.
        std::sort(tokill.begin(),tokill.end(),std::greater<int>());
        for (auto &i: tokill){
            del_agent(i);  // remove the agent from the population
        }
#ifdef DEBUG
        std::cout<<"Checking for zombies...";
        for(auto &a : population_) {
            assert(!get_agent_at(a).is_dead());
        }
        std::cout<<" no zombies found"<<std::endl;
#endif //DEBUG
    }

    void Population::cap_population(){
        // assumes population_.size() > nmax_
        std::uniform_real_distribution<double> dist01(0,1);
        double s=(double)population_.size();
        double p=std::min(1-(nmax_/s),1.0); // probability of killing, increases with population size
        int c=0;
        for(std::vector<population_type>::iterator it=population_.begin();it!=population_.end();it++){
            if(dist01(rng)<p*0.2){
                c++;
                fields_[get_field_at(it)].rem_agent(); // remove the agent from the field
                get_agent_at(it).set_energy(-1);
                del_agent(it);
            }
        }
#ifdef DEBUG
        std::cout << "Population is too large: "<<s<<">"<<nmax_<<"... killing agents with prob "<<p<<" for a total of "<<c<<".\n";
#endif
    }
    // TODO needs debugging
    void Population::debug_step(int fieldID, Genome::perception_type perceptions) {
        // check that the number of agents seen is correct
        std::vector<std::vector<int> > neighbors = field_of_view(fov_radius_,fieldID,field_size_,field_size_*field_size_); // indexes of neighbors
        std::vector<int>::iterator it;
        std::vector<int> sums (neighbors.size(),0);
        std::vector<int> foods (neighbors.size(),0);
        for(auto & b : population_) { // count agents
            for (size_t i=0;i<neighbors.size();i++) {
                it = std::find(neighbors[i].begin(),neighbors[i].end(),get_field_at(b));
                if(it!=neighbors[i].end()) // if the agent is in one of the visible cells
                    sums[i]++;
            }
        }
        sums[2]--;      // do not count the current agent in the current cell

        for(size_t i=0;i<neighbors.size();i++)
            for(auto &v:neighbors[i]){ // count food
                //XXXXFOOD
                foods[i]+=fields_[v].get_food(0);
            }
#ifdef INVISIBLE_FOOD
        assert(perceptions[2]==foods[2]); // check that the perceived food is real
#ifdef INTERACT
        for (size_t i = 0; i < neighbors.size(); i++)
            if(i!=2){                             // food here
                assert__(perceptions[i]==sums[i]) { // check that the perceived agents are real
                    std::vector<int> visible_cells = get_neighboring_cells(fov_radius_,fieldID,field_size_,field_size_*field_size_); // indexes of neighbors
                    std::cout<<"Cell "<<visible_cells[i]<<" contains "<<sums[i]<<" agents but "<<perceptions[i+1]<<" are perceived"<<std::endl;
                }}
#endif //INTERACT
#elif !defined INVISIBLE_FOOD
        for (size_t i = 0; i < neighbors.size(); i++) {
            assert(perceptions[i]==foods[i]); // check that the perceived food is real
#ifdef INTERACT
            assert(perceptions[i+neighbors.size()]==sums[i]); // check that the perceived agents are real
#endif //INTERACT
        }
#endif //INVISIBLE_FOOD
    }

    //! Create a new food source at a random location and with a random food quantity
    void Population::create_food_source(std::vector<field_type> &fields, /*!< The environment */
                                        int x0, /*!< The minimum food quantity */
                                        int x1, /*!< The maximum food quantity */
                                        int init_food,  /*!< The quantity of food present in the source to be replaced. Used only for bimodal distribution */
                                        size_t food_type, /*!< Type of food to be created */
                                        bool exclusive, /*!< True - only one type of food per field */
                                        bool lock_food
                                        ) {
        std::uniform_int_distribution<int> fields_index_distribution(0,fields.size()-1);
        std::uniform_int_distribution<int> initial_food_quantity(x0,x1);
        int inc=0;
        ///////////////////////////
        // Uniformly distributed //
        ///////////////////////////
#ifdef FOOD_UNIFORM
        // we generally use this
        inc=initial_food_quantity(rng);
#endif
        // TODO check if it makes any difference
        ///////////////
        // Power law //
        ///////////////
#ifdef FOOD_POWER               // TODO check for bugs
        std::uniform_real_distribution<double> dist01(0,1);
        double rand01=dist01(rng);
        double n = 0.01;
        inc=std::pow((std::pow(x1,n+1)-std::pow(x0,n+1))*rand01+std::pow(x0,n+1),1/(n+1));
#endif
        ///////////////////
        // Bimodal distr //
        ///////////////////
#ifdef FOOD_BIMODAL
        std::uniform_real_distribution<double> dist01(0,1);
        std::uniform_int_distribution<int> initial_food_quantity_low(1,5);
        std::uniform_int_distribution<int> initial_food_quantity_high(x1/2,x1);
        if(init_food==0) {      // there was no food previously, pick a type at random
            if(dist01(rng)<=FOOD_BIMODAL_PROB)
                inc=initial_food_quantity_low(rng);
            else
                inc=initial_food_quantity_high(rng);
        } else if(init_food<x1/2) // previously a poor field, spawn little food
            inc=initial_food_quantity_low(rng);
        else                    // previously a rich field, spawn much food
            inc=initial_food_quantity_high(rng);
#endif
        // ------------------------------
        //FOOD
        int max_search(500);//number of times it should try to find an empty cell
        int count(0); //current trial
        int idx=fields_index_distribution(rng);

        if(exclusive){
            while((fields[idx].get_all_food()!=0)&&(count<max_search)){ // find a new field if it already contains food
                idx=fields_index_distribution(rng);
                count++;
            }
        }else{
            while((fields[idx].get_food(food_type)!=0)&&(count<max_search)){ // find a new field if it already contains food
                idx=fields_index_distribution(rng);
                count++;
            }
        }
#ifdef DEBUG
        if(exclusive){
            assert(fields[idx].get_all_food()==0);
        }else{
            assert(fields[idx].get_food(food_type)==0);
        }

        std::cout<<": creating a new food source in field "<<idx<<" of value "<<inc<<" type "<<food_type<<std::endl;
        // WARNING: random selection might replenish the cell that was emptied in the current round, more often in small environments.
#endif //DEBUG
        //FOOD


        fields[idx].inc_food(food_type,inc);
        //FOOD
        fields[idx].set_initial_food(food_type,inc);

        if(lock_food)
            fields[idx].lock_food(food_type);
    }

void Population::remove_food_source(std::vector<field_type> &fields,size_t food_type){

        std::uniform_int_distribution<int> fields_index_distribution(0,fields.size()-1);
        size_t idx=fields_index_distribution(rng);

        for(size_t count=0;count<field_size_*field_size_;count++){ // find a new field if it already contains food
            if(fields[idx].get_food(food_type)!=0) break;
            idx++;
            if(idx>=fields.size()){
                idx=0;
            }
        }
#ifdef DEBUG
        assert(fields[idx].get_food(food_type)!=0);
        std::cout<<": deleting food source in field "<<idx<<" of value "<<fields[idx].get_food(food_type)<<" type "<<food_type<<std::endl;
#endif //DEBUG
        //FOOD
        fields[idx].set_food(food_type,0);
        //FOOD
        fields[idx].set_initial_food(food_type,0);

    }




} // end namespace
