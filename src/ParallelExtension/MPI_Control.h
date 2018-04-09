/*
 * Author: Leonel Aguilar
 *
 * Created on May 29, 2017, 6:59 AM
 */

//MPI
#include <mpi.h>
#include <iostream>
#include <memory>
#include <vector>

namespace PARALLEL_EXT{
    template <typename T> void E_Abort(T error) {
        std::cerr <<" ERROR: "<< error << std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    
    
    class comm_wrap{        
        MPI_Comm comm;
    public:
        comm_wrap():comm(MPI_COMM_NULL){};        
        comm_wrap(const MPI_Comm original_comm){
            if((original_comm==MPI_COMM_WORLD)||(original_comm==MPI_COMM_SELF)){
                MPI_Comm_dup(original_comm,&comm);
            }else{
                E_Abort("comm wrap can only be instantiated with comm types that do not required to be freed");
            }
        }
        
        comm_wrap(const comm_wrap &original_comm){
                MPI_Comm_dup(original_comm.show_comm(),&comm);
        }
        
        ~comm_wrap(){
            if(comm!=MPI_COMM_NULL){
                MPI_Comm_free(&comm); 
            }            
        }        
        void dup_comm(const MPI_Comm &original_comm){
            MPI_Comm_dup(original_comm,&comm);    
        }
        
        MPI_Comm show_comm() const {
            return comm;
        }
        
        MPI_Comm* show_comm_p(){
            return &comm;
        }
    };
    
    class Group_wrap{        
        MPI_Group group;
    public:
        Group_wrap():group(MPI_GROUP_NULL){};               
        Group_wrap(const MPI_Comm comm){
            MPI_Comm_group(comm,&group);
        }        
        ~Group_wrap(){
            if(group!=MPI_GROUP_NULL){
                MPI_Group_free(&group); 
            }            
        }                
        MPI_Group show_group() const {
            return group;
        }
    };
    
    //WIP
    class agent_partition{
        int part_ID;
        int agent_count;
    };
    
    class interCommControl{
        
        
        
        
    };
    
    
    class mpi_parallel_ext{
    private:
        
        //Up-level Membership
        comm_wrap uplevel_comm;
        int upl_size,upl_rank;
        
        //Group Membership
        Group_wrap sub_grp;
        comm_wrap  comm;
        int num_groups;
        int group_id;
        int size,rank; 
        
        //Inter-communicator to communicate with other processes
        comm_wrap parent_comm;
        int parent_rank;
        bool has_parent;
    
    public:   
       explicit mpi_parallel_ext(const comm_wrap &uplevel_comm_,int num_groups_):uplevel_comm(uplevel_comm_),upl_size(-1),upl_rank(-1),num_groups(num_groups_),group_id(-1),size(-1),rank(-1),parent_rank(-1),has_parent(false){
           MPI_Comm_rank(uplevel_comm.show_comm(), &upl_rank);	/* get upper level process id */
           MPI_Comm_size(uplevel_comm.show_comm(), &upl_size);	/* get upper level of processes */
           
           //CHECK Needs to be changed when moving to capability computing
           if (upl_size < num_groups_) {
                if (upl_rank == 0) std::cerr << " Not enough number of process, spawn #processes >= samples ("<<upl_size<<"!+"<<num_groups_<<")"<< std::endl;
                E_Abort(" Not enough number of process, spawn #processes >= samples ");
           }else if((upl_size%num_groups)!=0){
                if (upl_rank == 0) std::cerr << " WARNING un-homogeneous number of groups try #processes = samples*#cores_per_sample " << std::endl;
           }
      
           int group_size= ((float) upl_size) / (float) num_groups; // Determine color based on row
           group_id = ((float) upl_rank) / (float) group_size; // Determine color based on row
           
           std::cout<<" #upl_rank "<<upl_rank<<" num_groups "<<num_groups<<" Group ID "<<group_id<<" group_size "<<group_size<<std::endl;
           
           MPI_Comm_split(uplevel_comm.show_comm(),group_id,upl_rank,comm.show_comm_p());
           MPI_Comm_rank(comm.show_comm(), &rank);	/* get current process id */
           MPI_Comm_size(comm.show_comm(), &size);	/* get number of processes */
           
           CheckForParentProcess();
        }
       ~mpi_parallel_ext(){
          if(parent_comm.show_comm() != MPI_COMM_NULL){
                MPI_Barrier(parent_comm.show_comm());
                MPI_Comm_disconnect(parent_comm.show_comm_p());
          } 
       }
       
       
       void BarrierOnSimulation(){
           MPI_Barrier(uplevel_comm.show_comm());
       }
       
       void BarrierOnSample(){
           MPI_Barrier(comm.show_comm());
       }
       
       int num_samples(){
           return num_groups;
       }
       bool sim_root(){
           return (upl_rank==0);
       }
       int sim_elem_id(){
           return upl_rank;
       }
     
       bool sample_root(){
           return (rank==0);
       }       
       int sample_id(){
           return group_id;
       }
       int within_sample_id(){
           return rank;
       }
       
       void printInfo(){
           std::cout<<" Group Number "<<group_id<<std::endl;
           std::cout<<"Upper level " << upl_rank << " of " << upl_size <<std::endl;
           std::cout<<"My rank " << rank << " of " << size <<std::endl;
       }

       void CheckForParentProcess(){
            MPI_Comm_get_parent(parent_comm.show_comm_p());
            if (parent_comm.show_comm() == MPI_COMM_NULL){
                if(sim_root()==0) std::cout << " ### No parent process" << std::endl;
            }else{
                std::cout<<"Connected to parent process "<<std::endl;
            }
        }
       
       void PrintParameters(std::vector<float> &params){
           for(auto &a:params){
               std::cout<<a<<std::endl;
           }
       }
       
       void ReceiveNewParameters(float &mutation_size_skill,float &mutation_size_weights,int &season_length,int &max_age,double &life_bonus,int &food_max,int &tot_num_food) {
           int expected_params_state_size(7);
           std::vector<float> params(expected_params_state_size);
           if (parent_comm.show_comm() == MPI_COMM_NULL) {
                std::cout << std::endl << std::endl << " No one to receive parameters from" << std::endl;
                return;
           }else{
               std::cout << "### RECEIVING PARAMETERS "<<std::endl;
           }
           int num_parent_params(0);
           int init_type(0);
           
           MPI_Bcast(&init_type, 1, MPI_INT, 0,parent_comm.show_comm());
           MPI_Bcast(&num_parent_params, 1, MPI_INT, 0,parent_comm.show_comm());

           std::cout << "\nNumber of params"<<num_parent_params<<std::endl;
           if (num_parent_params != expected_params_state_size){
                E_Abort("### Mismatch in the number of parameters exchanged given ");
           }
           MPI_Bcast(&params[0], num_parent_params, MPI_FLOAT,0,parent_comm.show_comm());
           
           mutation_size_skill=params[0];
           mutation_size_weights=params[1];
           if(init_type==0){
                season_length=params[2];
           }
           max_age=params[3];
           life_bonus=params[4];
           food_max=params[5];
           tot_num_food=params[6];

           PrintParameters(params);
           return;
        }
       
        void SendScores(bool success,std::vector<float> &res_phen,std::vector<float> &res_SDphen,std::vector<float> &res_gen,std::vector<float> &res_SDgen) {
           if (parent_comm.show_comm() == MPI_COMM_NULL) {
                std::cout << std::endl << std::endl << " No one to send parameters to" << std::endl;
                return;
           }
           int scores_length(0);
           
           if(success){
               scores_length=res_phen.size();
           }
           
           int rbuf[2]={-1,-1}, recvcount(-1);
           int *displs(NULL),*rcounts(NULL);
           
           std::cout<<" child1 "<<scores_length<<std::endl;
           
           MPI_Gather(&scores_length, 1, MPI_INT,&rbuf,recvcount, MPI_INT,0,parent_comm.show_comm());
           
           MPI_Gatherv(&res_phen[0],scores_length, MPI_FLOAT, &rbuf, rcounts,displs, MPI_FLOAT, 0,parent_comm.show_comm());
           MPI_Gatherv(&res_SDphen[0],scores_length, MPI_FLOAT, &rbuf, rcounts,displs, MPI_FLOAT, 0,parent_comm.show_comm());
           MPI_Gatherv(&res_gen[0],scores_length, MPI_FLOAT, &rbuf, rcounts,displs, MPI_FLOAT, 0,parent_comm.show_comm());
           MPI_Gatherv(&res_SDgen[0],scores_length, MPI_FLOAT, &rbuf, rcounts,displs, MPI_FLOAT, 0,parent_comm.show_comm());
           
           std::cout<<" FINISHED SENDING "<<std::endl;
           return;
        }         
    };
}