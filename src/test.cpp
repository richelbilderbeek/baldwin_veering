//#include "tiny_cnn/tiny_cnn.h"
#include "genome.hpp"
#include <random>
#include <unordered_map>
#include <iomanip>              // setiosflags
#include "sstream"
#include <unistd.h>
#include <mpi.h>

//using namespace tiny_cnn;
//using namespace tiny_cnn::activation;

typedef std::vector<double> cv_type;
typedef std::vector<double> vec_t;
typedef long int label_t;

template<typename T1,typename T2>
class Experience{
public:
    T1 state;
    int state_idx;
    T2 action;
    float reward;
    T1 state_next;
    int state_next_idx;

    Experience(T1 state_, int state_idx_, T2 action_,float reward_,T1 state_next_, int state_next_idx_):
        state(state_), state_idx(state_idx_), action(action_),reward(reward_),state_next(state_next_),state_next_idx(state_next_idx_)
    {}
};

void construct_cnn();
void random_walk();
void ann_walk();
void test_gen(Joleste::Genome &gen,int ID);
void exec_gen(Joleste::Genome &gen,cv_type &current_state,float &reward,label_t &action,std::vector<Experience<cv_type,label_t> > &memory,std::unordered_map<std::string,int> &seen_states,int timestep,int ID);
void gen_walk();
bool add_experience(int start,int end,std::unordered_map<std::string,int> &seen_states);
long seedgen(int ID);
void log_test_gen(Joleste::Genome &gen,int ID,std::ostringstream &str_stats);
void log_INIT(std::ofstream &statsFile, const std::string &fileName, std::vector<std::string> &colnames);
void log_WRITE(std::ostringstream &str_stats, std::ofstream &statsFile);
void log_SIMPLE_INIT(const std::string &fileName,std::ofstream &statsFile);
    
int main(int argc, const char** argv){
    MPI_Init(&argc, const_cast<char***>(&argv));
    std::cout<<"##Starting Construction"<<std::endl;
    //construct_cnn();
    //random_walk();
    //ann_walk();
    long seed=seedgen(0);
    srand(seed);
    Joleste::rng.seed(seed);
    gen_walk();
    std::cout<<"##Ended Construction"<<std::endl;
    MPI_Finalize();
}

float MaxQVal(vec_t q_values){
    float maxq=q_values[0];
    for(size_t i=1;i<q_values.size();i++){
        if(q_values[i]>maxq) maxq=q_values[i];
    }
    return maxq;
}

//vec_t create_state(float a, float b, float c, float d, float e){
//    vec_t state;
//    state.push_back(a);
//    state.push_back(b);
//    state.push_back(c);
//    state.push_back(d);
//    state.push_back(e);
//    return state;
//}

std::vector<double> create_state_compat(float a, float b, float c, float d, float e){
    std::vector<double> state;
    state.push_back(a);
    state.push_back(b);
    state.push_back(c);
    state.push_back(d);
    state.push_back(e);
    return state;
}

template<typename T1>
void print_state(T1 state, bool endline=true){
    //std::cout<<std::endl;
    for(auto v:state){
        std::cout<< std::setiosflags(std::ios::fixed) << std::setprecision(2) <<v<<" ";
    }
    if(endline)
     std::cout<<std::endl;
}

double calculate_error(std::vector<double> obs,std::vector<double> real){
    assert(obs.size()==real.size());
    double error(0);
    for(size_t i=0;i<obs.size();i++){
        error+=fabs(obs[i]-real[i])/real[i];
    }
    return error;
}

template<typename T1>
void pprint_state(T1 state){
    std::cout<<"# ";
    for(auto v:state){
        if(std::abs(v-1.0)<0.01){
            std::cout<<"X";
        }else std::cout<<"_";
    }
    std::cout<<std::endl;
}

void pprint_action(int action){
        switch (action) {
        case 0 :
          // Moving left
          std::cout<<" <- "<<std::endl;
          break;
        case 1 :
          // Moving right
          std::cout<<" -> "<<std::endl;
          break;
        case 2 :
          //eat
          std::cout<<" ^EAT^ "<<std::endl;
          break;
        default :
          // Process for all other cases.
          std::cout<<" Invalid action "<<(int) action<<std::endl;
    };
}

template<typename T1,typename T2>
void transition_state(T1 state,T2 action,float &reward, int &nid,T1 &next_state,std::vector<Experience<T1,T2> > &memory,std::unordered_map<std::string,int> &seen_states){
    //std::cout<<std::endl;
    T1 out_state(5,0);
    int sidx(0);   
    reward=0;
    for(auto v:state){
        //std::cout<<v<<" ";
        if(v == 1.0) break;
        sidx++;
    }
    
#if defined(ACTION_REPLAY)    
    int start_idx(0);
    start_idx=sidx;
#endif
    //std::cout<<"CURRENT ID "<<sidx<<std::endl;

    switch (action) {
        case 0 :
          // Moving left
          //std::cout<<"moving left"<<std::endl;
          if(sidx>0) sidx--;
          //else reward=-1;
          break;
        case 1 :
          // Moving right
          //std::cout<<"moving right"<<std::endl;
          if(sidx<4) sidx++;
          //else reward=-1;
          break;
        case 2 :
          //eat
          //std::cout<<" EATING "<<std::endl;
          if(sidx==2)
            reward=10;
          //else reward=-1;
          break;
        default :
          // Process for all other cases.
          std::cout<<" Invalid action "<<(int) action<<std::endl;
    }
    out_state[sidx]=1;
    next_state=out_state;
    nid=sidx;
//    //Staticreward CHECK!
//    if(add_experience(start_idx,nid,seen_states))
//        memory.push_back(Experience<T1,T2>(state,start_idx,action,reward,next_state,sidx));
}


vec_t create_action(float a, float b, float c){
    vec_t state;
    state.push_back(a);
    state.push_back(b);
    state.push_back(c);
    return state;
}

std::vector<double>  create_action_compat(float a, float b, float c){
    std::vector<double>  state;
    state.push_back(a);
    state.push_back(b);
    state.push_back(c);
    return state;
}

void exec_gen(Joleste::Genome &gen,cv_type &current_state,float &reward,label_t &action,std::vector<Experience<cv_type,label_t> > &memory,std::unordered_map<std::string,int> &seen_states,int timestep,int ID){
    // cv_type temp_qval;
    int nextid;
    cv_type next_state;
    //std::cout<<"#### "<<ID<<"Timestep # "<<timestep<<std::endl;
    gen.train(current_state,reward,timestep);
    // TODO clean up and make consistent with code
    action=gen.decision_algorithm(current_state);
    transition_state(current_state,action,reward,nextid,next_state,memory,seen_states);
    //pprint_action(action);
    //pprint_state(next_state);
    current_state=next_state;
}

//#ifdef BRAIN_PQL
//#elif defined(BRAIN_DEEP)
//#endif

void gen_walk() {
      
    int MaxSteps(5000);
    //Agent 0
    
    int size,rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::ostringstream str_stats;
    std::ofstream statsFile;
    double t1, t2,t3;
    std::ostringstream fileName;
    
    fileName<<"Agent_";
#if defined(BRAIN_QL)
    fileName<<"QL_";
#elif defined(BRAIN_DEEP)
    fileName<<"DRL_";
#elif defined(BRAIN_RQL)
    fileName<<"RQL_";
#elif defined(BRAIN_PQL)
    fileName<<"PQL_";
#else
    fileName<<"NOBRAIN_";
#endif

#if defined(EPOCHS)    
    fileName<<"EP_"<<EPOCHS<<"_";
#else
    fileName<<"EP_1_";
#endif
    
#if defined(AREPLAY)    
    fileName<<"AR_"<<AREPLAY<<"_";
#else
    fileName<<"AR_0_";  
#endif
    fileName<<rank<<".csv";
        
    log_SIMPLE_INIT(fileName.str(),statsFile);
    
    Joleste::Genome gen_0;
    //gen_0.brain_.ann.save_weights("gen_0_START");
    cv_type current_state_0(create_state_compat(0.,0.,1.,0.,0.));
    float reward_0(0);
    label_t action_0(2);
    std::vector<Experience<cv_type,label_t> > memory_0;
    std::unordered_map<std::string,int> seen_states_0;
    //print_state(current_state_0);
    //pprint_state(current_state_0);
    for(int i=0;i<MaxSteps;i++){
        str_stats<<i<<",";
        t1 = MPI_Wtime();    
        exec_gen(gen_0,current_state_0,reward_0,action_0,memory_0,seen_states_0,i,0);
        t2 = MPI_Wtime();
        log_test_gen(gen_0,0,str_stats);
        t3 = MPI_Wtime();
        str_stats<<action_0<<","<<reward_0<<","<<(t2-t1)<<","<<(t3-t2)<<std::endl;
    }
    log_WRITE(str_stats,statsFile);
    if(statsFile.is_open()) statsFile.close();
    //test_gen(gen_0,0);
}
void log_test_gen(Joleste::Genome &gen,int ID,std::ostringstream &str_stats){
    std::vector<cv_type> test_states;
    std::vector<vec_t> result_states;
    std::vector<int>result_action;
    test_states.push_back(create_state_compat(1,0,0,0,0));
    test_states.push_back(create_state_compat(0,1,0,0,0));
    test_states.push_back(create_state_compat(0,0,1,0,0));
    test_states.push_back(create_state_compat(0,0,0,1,0));
    test_states.push_back(create_state_compat(0,0,0,0,1));
    result_states.push_back(create_action(0.0111111,0.1111111,0.0111111));
    result_states.push_back(create_action(0.0111111,1.1111111,0.1111111));
    result_states.push_back(create_action(0.1111111,0.1111111,11.1111111));
    result_states.push_back(create_action(1.1111111,0.0111111,0.1111111));
    result_states.push_back(create_action(0.1111111,0.0111111,0.0111111)); 
    result_action.push_back(1);
    result_action.push_back(1);
    result_action.push_back(2);
    result_action.push_back(0);
    result_action.push_back(0);
  
    double total_error(0),error(0);
    std::cout<<"------------------------------"<<std::endl;
    for(size_t i=0;i<test_states.size();i++){
        cv_type temp_qval=gen.activate(test_states[i]);
        print_state(temp_qval,false);
        error=calculate_error(temp_qval,result_states[i]);
        str_stats<<error<<",";        
        std::cout<<" | "<<error<<std::endl;
        total_error+=calculate_error(temp_qval,result_states[i]);
    }
    std::cout<<"###############################"<<std::endl;
    str_stats<<total_error<<","; 
    
    std::cout<<"TOTAL  err = "<<total_error<<std::endl;
  
    //std::cout<<" ACTION "<<std::endl;
    int test_action(0);
    int total_actions(0);
    for(size_t i=0;i<test_states.size();i++){
        test_action=gen.best_action(test_states[i]); //brain_.make_choice(test_states[i],true);
        //std::cout<<test_action<<std::endl;
        if(test_action==result_action[i])
            total_actions+=1;
    }
    //std::cout<<" Correct actions "<<total_actions<<std::endl;
    str_stats<<total_actions<<","; 
}


void test_gen(Joleste::Genome &gen,int ID){
    std::vector<cv_type> test_states;
    std::vector<vec_t> result_states;
    std::vector<int>result_action;
    test_states.push_back(create_state_compat(1,0,0,0,0));
    test_states.push_back(create_state_compat(0,1,0,0,0));
    test_states.push_back(create_state_compat(0,0,1,0,0));
    test_states.push_back(create_state_compat(0,0,0,1,0));
    test_states.push_back(create_state_compat(0,0,0,0,1));
    result_states.push_back(create_action(0.0111111,0.1111111,0.0111111));
    result_states.push_back(create_action(0.0111111,1.1111111,0.1111111));
    result_states.push_back(create_action(0.1111111,0.1111111,11.1111111));
    result_states.push_back(create_action(1.1111111,0.0111111,0.1111111));
    result_states.push_back(create_action(0.1111111,0.0111111,0.0111111)); 
    result_action.push_back(1);
    result_action.push_back(1);
    result_action.push_back(2);
    result_action.push_back(0);
    result_action.push_back(0);
  
    std::cout<<"### "<<ID<<" TEST 1"<<std::endl;
    double total_error(0);
    for(size_t i=0;i<test_states.size();i++){
        cv_type temp_qval=gen.activate(test_states[i]);
        print_state(temp_qval,false);
        std::cout<<" "<<calculate_error(temp_qval,result_states[i])<<std::endl;
        total_error+=calculate_error(temp_qval,result_states[i]);
    }
    
    std::cout<<"TOTAL  err = "<<total_error<<std::endl;
  
    std::cout<<" ACTION "<<std::endl;
    int test_action(0);
    int total_actions(0);
    for(size_t i=0;i<test_states.size();i++){
        test_action=gen.best_action(test_states[i]); //brain_.make_choice(test_states[i],true);
        std::cout<<test_action<<std::endl;
        if(test_action==result_action[i])
            total_actions+=1;
    }
    std::cout<<" Correct actions "<<total_actions<<std::endl;
}


bool add_experience(int start,int end,std::unordered_map<std::string,int> &seen_states){
      std::stringstream ss1,ss2;
            ss1.clear();
            ss1.str("");
            ss1<<start<<"_"<<end;

            if(seen_states.count(ss1.str())){
                std::cout<<"Already seen "<<ss1.str()<<std::endl;
                return false;
            }else{
               seen_states[ss1.str()]=14;
               return true;
            }
            return false;
}
//Random seed generator from Helmut G. Katzgraber, Random Numbers in Scientific Computing: An Introduction, arXiv:1005.4117v1
long seedgen(int ID){
    long s, seed, pid;
    pid = getpid();
    //std::cout<<"PID "<<pid<<std::endl;
    s = ID+time(NULL); /* get CPU seconds since 01/01/1970 */
    seed = labs(((s*181)*((pid-83)*359))%104729);
    return seed;
}

void log_INIT(std::ofstream &statsFile, const std::string &fileName, std::vector<std::string> &colnames) {
        std::ostringstream str_stat; // for output
        if (!fileName.empty()) {
            statsFile.open(fileName); //Opening file to print info to
            for (size_t i=0;i<colnames.size();i++){
                str_stat << colnames[i];
                if(i != colnames.size()-1)
                    str_stat<<",";
            }
            str_stat << std::endl;
            statsFile << str_stat.str();
            str_stat.str(std::string());
        }
    }

void log_WRITE(std::ostringstream &str_stats, std::ofstream &statsFile) {
        if (statsFile.is_open()) {
            statsFile << str_stats.str();
            statsFile.flush(); // we are going to delete the string in a moment
        }
    }

void log_SIMPLE_INIT(const std::string &fileName,std::ofstream &statsFile) {
        // initialize output file
        // When changing this column names, change as well the order of output of log_env_stats())
        std::vector<std::string> colnames = {"timeStep","1","2","3","4","5","TOT","CORR","ACT","REWARD","TIME","T_TEST"};
        log_INIT(statsFile, fileName, colnames);
}