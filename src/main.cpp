//! Compile flags
/*!
  Compile FLAGS:
  DEBUG if enabled performs consistency checks and prints more output
  INTERACT if enabled makes agents aware of other agents
  INVISIBLE_FOOD if enabled agents can only see food in their current cell
  BINARY_IN if enabled inputs are binary, if disabled they increase with the number of food/agents
  FILTER_STATIC_PERCEPTIONS if enabled only agents that are static (eating) are counted when computing the perceptions
*/

#include "population.hpp"
#include <iostream>
#include <stdlib.h>
#include <stdexcept>

//For debugging
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

//MPI
#include <mpi.h>
#include "ParallelExtension/MPI_Control.h"

// option parser
#include "argparse.hh"

#include <fenv.h>

#ifdef _OPENMP
#include <omp.h>
#endif
//========= some macros for nicer presentation (not essential) =========
//use as litte macros as possible in c++ (most stuff can be solved without)
#define TYPE(T) typeid(T).name()
#define CLR_SCR() std::cout << "\033[2J\033[100A";
#define NEW_LINE() std::cout << std::endl;
#define WAIT_FOR_INPUT() while(std::cin.gcount() == 0) std::cin.get(); std::cin.get();
#define ASSERT_MSG(cond, msg) if(cond) {PRINT_RED(msg); throw std::runtime_error("error");}
#define PRINT_NAMED(x) std::cout << #x << " = " << x << std::endl; //#x changes the variable name into a string "x"
#define PRINT_RED(x) std::cout << "\033[1;31m" << x << "\033[0m" << std::endl;
#define PRINT_BLUE(x) std::cout << "\033[1;34m" << x << "\033[0m" << std::endl;
#define PRINT_CYAN(x) std::cout << "\033[1;36m" << x << "\033[0m" << std::endl;
#define PRINT_GREEN(x) std::cout << "\033[1;32m" << x << "\033[0m" << std::endl;
#define PRINT_YELLOW(x) std::cout << "\033[1;33m" << x << "\033[0m" << std::endl;
#define PRINT_MAGENTA(x) std::cout << "\033[1;35m" << x << "\033[0m" << std::endl;

void print_info();
void handler(int sig);
void clean_vis();
void clean_results(std::stringstream &filenameAgents,std::stringstream &filenameEnvironment,std::stringstream &filenameReprod);
void clean_all_results();
long seedgen(int);

using namespace Joleste;

int main(int argc, const char** argv) {
    int rank(0), world_size(0);
    /* starts MPI */
    MPI_Init(&argc, const_cast<char***>(&argv));
    {
        feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);  // Enable all floating point exceptions but FE_INEXACT
        ArgumentParser parser;
        //integer
        parser.addArgument("-n","--pop-size", 1 ,false);
        parser.addArgument("-f","--food-num",1,false);
        parser.addArgument("-s","--season-len",1,false);
        parser.addArgument("-S","--samples", 1,false);
        parser.addArgument("-l","--sim-len",1,false);
        parser.addArgument("-v","--fov",1,false);
        //floating point
        parser.addArgument("-F","--food-prop", 1,false);
        parser.addArgument("-P","--skill-lvl",1,false);
        //optional integer
        parser.addArgument("--field-size",1);
        parser.addArgument("--food-energy",1);
        parser.addArgument("--max-pop",1);
        parser.addArgument("--food-qty",1);
        parser.addArgument("--max-age",1);
        parser.addArgument("--seed-iter",1);
        parser.addArgument("--famine-iter",1);
        // optional integer array
        parser.addArgument("--save-pop",'+',true);
        // optional floating point
        parser.addArgument("--social-ratio",1);
        parser.addArgument("--life-bonus", 1);
        //optional strings
        parser.addArgument("--load-pop",1);
        parser.addArgument("--load-logic",1);

        parser.parse(argc,argv);

        std::size_t endptr=0;
        int num_agents=stoi(parser.retrieve<std::string>("pop-size"),&endptr,10);
        int tot_num_food=stoi(parser.retrieve<std::string>("food-num"),&endptr,10);
        int season_length=stoi(parser.retrieve<std::string>("season-len"),&endptr,10);
        int samples=stoi(parser.retrieve<std::string>("samples"),&endptr,10);
        int sim_length=stoi(parser.retrieve<std::string>("sim-len"),&endptr,10);
        int fov_radius=stoi(parser.retrieve<std::string>("fov"),&endptr,10);
        double food_proportion=stod(parser.retrieve<std::string>("food-prop"),&endptr);
        double skill_lvl_0=stod(parser.retrieve<std::string>("skill-lvl"),&endptr);
        double food_energy=parser.count("food-energy")>0 ? stod(parser.retrieve<std::string>("food-energy"),&endptr) : 1.0;
        int field_size=parser.count("field-size")>0 ? stoi(parser.retrieve<std::string>("field-size"),&endptr,10) : 20;
        int max_agents=parser.count("max-pop")>0 ? stoi(parser.retrieve<std::string>("max-pop"),&endptr,10) : 5000;
        int food_max=parser.count("food-qty")>0 ? stoi(parser.retrieve<std::string>("food-qty"),&endptr,10) : 200;
        int max_age=parser.count("max-age")>0 ? stoi(parser.retrieve<std::string>("max-age"),&endptr,10) : 1000;
        int seed_iteration=parser.count("seed-iter")>0 ? stoi(parser.retrieve<std::string>("seed-iter"),&endptr,10) : -1;
        int famine_iteration=parser.count("famine-iter")>0 ? stoi(parser.retrieve<std::string>("famine-iter"),&endptr,10): -1;
        std::vector<std::string> save_iters=parser.count("save-pop")>0 ? parser.retrieve<std::vector<std::string>>("save-pop") : std::vector<std::string>();
        double social_ratio=parser.count("social-ratio")>0 ? stod(parser.retrieve<std::string>("social-ratio"),&endptr) : 0.0;
        double life_bonus=parser.count("life-bonus")>0 ? stod(parser.retrieve<std::string>("life-bonus"),&endptr) : 1.0;
        std::string load_pop=parser.count("load-pop")>0 ? parser.retrieve<std::string>("load-pop") : "";
        std::string load_logic=parser.count("load-logic")>0 ? parser.retrieve<std::string>("load-logic") : "";
        int min_agents=10;
        int binary_in=1;
        int direct_feedback=0;
        //double food_energy=0;
        double antisocial_ratio=0.0;
        std::vector<size_t> save_pop;
        for(auto &a: save_iters) {
            save_pop.push_back(stoi(a,&endptr,10));
        }
        float mutation_size_skill(0.05);
        float mutation_size_weights(0.05);

    #ifdef DEBUG
        PRINT_RED("Starting tests");
    #endif

        PARALLEL_EXT::mpi_parallel_ext para(MPI_COMM_WORLD,samples);

        para.ReceiveNewParameters(mutation_size_skill,mutation_size_weights,season_length,max_age,life_bonus,food_max,tot_num_food);


        //    const double temperatur   = std::stod(argv[16]);
        //    const int const_mut_rate  = std::stoi(argv[17]);
        if(para.sim_root()) {
            PRINT_BLUE("Parameters are:");
            PRINT_GREEN("field_size: "<<field_size);
            PRINT_GREEN("num_agents: "<<num_agents);
            PRINT_GREEN("max_agents      : "<<max_agents);
            PRINT_GREEN("min_agent      : "<<min_agents);
            PRINT_GREEN("tot_num_food  : "<<tot_num_food);
            PRINT_GREEN("food_proportion  : "<<food_proportion);
            PRINT_GREEN("season_length  : "<<season_length);
            PRINT_GREEN("food_max      : "<<food_max);
            PRINT_GREEN("sim_length  : "<<sim_length);
            PRINT_GREEN("samples   : "<<samples);
            // PRINT_GREEN("temperatur: "<<temperatur);
            PRINT_GREEN("max_age   : "<<max_age);
            PRINT_GREEN("fov_radius: "<<fov_radius);
            PRINT_GREEN("binary_in : "<<binary_in);
            // PRINT_GREEN("const_mut_rate: "<<const_mut_rate);
            PRINT_GREEN("direct_feedback: "<<direct_feedback);
                PRINT_GREEN("skill0_level: "<<skill_lvl_0);
            PRINT_GREEN("food_energy: "<<food_energy);
            PRINT_GREEN("social_ratio: "<<social_ratio);
            PRINT_GREEN("antisocial_ratio: "<<antisocial_ratio);
            PRINT_GREEN("life_bonus: "<<life_bonus);
            PRINT_GREEN("seed_iteration: "<<seed_iteration);
            PRINT_GREEN("famine_iteration: "<<famine_iteration);
            PRINT_GREEN("save_iters:");
            for(auto &a:save_iters){
                std::cout << a<<" ";
            }
            std::cout << "\n";
            PRINT_GREEN("load_pop: "<<load_pop);
            PRINT_GREEN("load_logic: "<<load_logic);
            PRINT_GREEN("mutation_size_skill: "<<mutation_size_skill);
            PRINT_GREEN("mutation_size_weights: "<<mutation_size_weights);
        }

        print_info();

        int run_no(0);

        // std::stringstream out;
        // out << "agent.id,energy,seed" << std::endl;

        run_no++;
        std::stringstream filename_agents;
        std::stringstream filename_agents_gen;
        std::stringstream filename_environment;
        std::stringstream filename_reprod;
        std::stringstream filename_forage;
        std::stringstream filename_report;
        std::stringstream filename_pop;
        filename_report << "./results/report" << para.sample_id() << ".csv";
        filename_agents << "./results/phen_stats_agents_" << para.sample_id()  << ".csv";
        filename_agents_gen << "./results/stats_agents_" << para.sample_id()  << ".csv";
        filename_environment << "./results/stats_env_" << para.sample_id()  << ".csv";
        filename_reprod << "./results/stats_reprod_" << para.sample_id()  << ".csv";
        filename_forage << "./results/stats_forage_" << para.sample_id()  << ".csv";
        filename_pop << "./results/pop_dump_";
        PRINT_MAGENTA("Running sample " << rank+1 << " of " << world_size << " Saving results in " << filename_report.str() << " , " << filename_agents.str() << " , " << filename_agents_gen.str()<<" , "<<filename_environment.str()<<" , "<<filename_reprod.str()<<" , "<<filename_forage.str()<<" , "<<filename_pop.str());

        if(para.sim_root()){
            clean_all_results();
            clean_vis();
        }
        para.BarrierOnSimulation();
    // #ifdef DEBUG
    //     srand(rank);
    // #else
    //     srand(seedgen()); // time(NULL) after debugging
    //     //srand(0); // time(NULL) after debugging
    // #endif
    #ifdef DEBUG
            long seed=para.sim_elem_id();
    #else
            long seed=seedgen(para.sim_elem_id());
    #endif
            std::cout<<"Seeding process "<<para.within_sample_id()<<" with seed "<<seed<<std::endl;
            // srand(seed);
            rng.seed(seed);

        if (fov_radius*2+1 > field_size){
            if (para.sim_root()) std::cerr << "# Warning visual radius larger than the domain\n reducing visual radius to domain size" << std::endl;
            fov_radius = (int)(((float)field_size - 1.0)/2.0); //if domain size is even, a column and row will be blinded
        }

        if (tot_num_food > field_size*field_size){
            if (para.sim_root()) std::cerr << "# Warning more food than cells\n reducing number of food" << std::endl;
            tot_num_food = field_size*field_size; //if domain size is even, a column and row will be blinded
        }


        Population pop(para.sample_id(),max_agents, min_agents, num_agents, tot_num_food,food_proportion,season_length,mutation_size_skill,mutation_size_weights, food_max, field_size, max_age, fov_radius, binary_in,direct_feedback, skill_lvl_0, food_energy,life_bonus,seed_iteration,famine_iteration,save_pop,load_pop,load_logic);
        double start=0,end=0;
        start= MPI_Wtime();

        std::vector<float> res_phen;
        std::vector<float> res_SDphen;
        std::vector<float> res_gen;
        std::vector<float> res_SDgen;
        bool success(false);

        pop.simulate(res_phen,res_SDphen,res_gen,res_SDgen,sim_length, social_ratio, antisocial_ratio, filename_agents.str(), filename_agents_gen.str(),filename_environment.str(),filename_reprod.str(),filename_forage.str(),filename_pop.str()); // change the name of the file or have it write always in the same file

        if(res_phen.size()!=(uint) sim_length){
            std::cout<<"### ERROR:: there was an error in the simulation "<<std::endl;
        }else{
            success=true;
        }
        para.SendScores(success,res_phen,res_SDphen,res_gen,res_SDgen);

        end= MPI_Wtime();
        PRINT_RED("# " << para.sample_id() << " Sample finished... Time = "<<(end-start));

        // std::ofstream report;
        // report.open(filename_report.str()); //Opening file to print info to
        // report << out.str();
        // report.close();

    }
    MPI_Finalize();
    return 0;
}

void print_info(){
      int numprocs, rank, namelen;
      char processor_name[MPI_MAX_PROCESSOR_NAME];
      int iam = 0, np = 1;

      MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Get_processor_name(processor_name, &namelen);
#ifdef _OPENMP
      #pragma omp parallel default(shared) private(iam, np)
      {
        np = omp_get_num_threads();
        iam = omp_get_thread_num();
        printf("Thread %d out of %d from process %d out of %d on %s\n",
               iam, np, rank, numprocs, processor_name);
      }
#else
    printf("Process %d out of %d on %s\n",
               rank, numprocs, processor_name);

#endif
}

void handler(int sig) {
    void *array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

void clean_vis() {
    int ret(0);
    ret = system("rm -rf vis");
    if (ret)
        std::cerr << " System return vaule is " << ret << std::endl;
    ret = system("mkdir vis");
    if (ret)
        std::cerr << " System return vaule is " << ret << std::endl;
}

void clean_all_results() {
    int ret(0);
    ret = system("rm -rf results");
    if (ret)
        std::cerr << " System return value is " << ret << std::endl;
    ret = system("mkdir results");
    if (ret)
        std::cerr << " System return value is " << ret << std::endl;
}

void clean_results(std::stringstream &filenameAgents,std::stringstream &filenameEnvironment,std::stringstream &filenameReprod) {
    /// if files written remove them when repeated
    char buffer[200];
    //sprintf(buffer, "sh ./remove.sh %s", filename); /// remove.sh should take the filename and remove it
    if ((!filenameAgents.str().empty())&&(!filenameEnvironment.str().empty())&&(!filenameReprod.str().empty())) {
        sprintf(buffer, "rm %s %s %s", filenameAgents.str().c_str(),filenameEnvironment.str().c_str(),filenameReprod.str().c_str()); /// remove.sh should take the filename and remove it
        int ret(0);
        ret = system(buffer); /// should remove all old files
        if (ret)
            std::cerr << " System return value is " << ret << std::endl;
    }
}
//Random seed generator from Helmut G. Katzgraber, Random Numbers in Scientific Computing: An Introduction, arXiv:1005.4117v1
long seedgen(int ID){
    long s, seed, pid;
    pid = getpid();
    s = ID+time(NULL); /* get CPU seconds since 01/01/1970 */
    seed = labs(((s*181)*((pid-83)*359))%104729);
    return seed;
}

