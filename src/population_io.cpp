#include "population.hpp"
#include <cereal/archives/xml.hpp>
#include "cereal/types/vector.hpp"
#include <cereal/types/utility.hpp>

namespace Joleste {

    void Population::log_INIT(std::ofstream &statsFile, const std::string &fileName, std::vector<std::string> &colnames) {
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

    void Population::log_WRITE(std::ostringstream &str_stats, std::ofstream &statsFile) {
        if (statsFile.is_open()) {
            statsFile << str_stats.str();
            statsFile.flush(); // we are going to delete the string in a moment
        }
    }

    void Population::log_env_stats_INIT(const std::string &fileName) {
        // initialize output file
        // When changing this column names, change as well the order of output of log_env_stats())
        std::vector<std::string> colnames = {"timeStep","x","y","food0","food1","locked0","locked1","agents_in_cell","times_unlocked0","times_unlocked1","times_shared0","times_shared1"};
        log_INIT(statsFileEnv, fileName, colnames);
    }

    void Population::log_forage_stats_INIT(const std::string &fileName) {
        // initialize output file
        // When changing this column names, change as well the order of output of log_env_stats())
        std::vector<std::string> colnames = {"timeStep","ID","success","food_type","skill_0","skill_1","skill_0_gen","skill_1_gen"};
        log_INIT(statsFileForage, fileName, colnames);
    }

    //LOG

    void Population::log_forage_stats(Agent &agent, const size_t &timeStep, const bool success, const size_t food_type, std::ostringstream &str_stats) {
        str_stats << timeStep << "," << agent.get_ID() << "," << success<<","<<food_type<<","<< agent.get_skill(0) << "," << agent.get_skill(1)<< ","<< agent.get_skill_gen(0) << "," << agent.get_skill_gen(1) << std::endl;
    }

    void Population::log_env_stats(std::ostringstream &str_stats, const size_t &timeStep) {
        // increase food in fields
        int food[FOOD_TYPES];
        int locked[FOOD_TYPES];
        int times_shared[FOOD_TYPES];
        int times_unlocked[FOOD_TYPES];
        int agents_in_cell(0);
        bool write_cell(false);

        for (size_t i = 0; i < fields_.size(); i++) {
            //FOOD (CHECK)
            agents_in_cell=fields_[i].get_num_agents();
            write_cell=false;
            if(agents_in_cell)
                write_cell=true;
            for (size_t idx = 0; idx < FOOD_TYPES; idx++) {
                food[idx] = fields_[i].get_food(idx);
                locked[idx]=fields_[i].is_food_unlocked(idx);
                times_unlocked[idx]=fields_[i].get_times_unlocked(idx);
                times_shared[idx]=fields_[i].get_times_shared(idx);
                // ---------- log information about fields ----------
                if ((food[idx] > 0)||(times_unlocked[idx])||(times_shared[idx])){
                    write_cell=true;
                }
                //Reset the additional values
                fields_[i].reset_times_unlocked(idx);
                fields_[i].reset_times_shared(idx);
            }

            if(write_cell){
                int x = i % field_size_;
                int y = (int) i / field_size_;
                str_stats<< timeStep<<"," << x<<"," << y<<"," << food[0]<<"," << food[1]<<"," << locked[0]<<"," << locked[1]<<"," << agents_in_cell<<","<<times_unlocked[0]<<","<<times_unlocked[1]<<","<<times_shared[0]<<","<<times_shared[1]<< std::endl;
            }
        }
    }

    void Population::log_agent_stats_INIT(const std::string &fileName,std::ofstream &statsFile) {
        // initialize output file
        // When changing this column names, change as well the order of output of log_agent_stats())
        std::vector<std::string> colnames = {"timeStep","ID","x","y","energy","action","seed","skill_0","skill_1","skill_0_gen","skill_1_gen","age"};
        std::vector<std::string> cols=visualsys.names;
        for (auto&n : cols) {
            colnames.push_back(n);
        }
        log_INIT(statsFile, fileName, colnames);
    }

    //LOG

    void Population::log_agent_stats(const Agent &agent, const int fieldID, const Genome::actions_type &result, const size_t &timeStep, std::ostringstream &str_stats)const{
        int x = fieldID % field_size_;
        int y = fieldID / field_size_;
        str_stats << timeStep << "," << agent.get_ID() << "," << x << "," << y << "," << agent.get_energy() << "," << agent.get_action() << "," << agent.get_seed() << "," << agent.get_skill(0) << "," << agent.get_skill(1)<< ","<< agent.get_skill_gen(0) << "," << agent.get_skill_gen(1)<< ","<<agent.get_age();
        assert(!result.empty());
        for (unsigned int i = 0; i < result.size(); i++) {
            str_stats << "," << result[i];
        }
        str_stats << std::endl;
    }

    void Population::log_reprod_stats_INIT(const std::string &fileName) {
        // initialize output file
        // When changing this column names, change as well the order of output of log_agent_stats())
        std::vector<std::string> colnames = {"timeStep","ID","parentID","skill_0","skill_1","skill_0_gen","skill_1_gen"};
        for (auto &p:visualsys.names) {
            for (size_t i = 0; i < N_OUTPUTS; i++) {
                colnames.push_back(std::string(p+"_"+Agent::ntoa(i)));
            }
        }
        log_INIT(statsFileReprod, fileName, colnames);
    }

    //LOG

    void Population::log_reprod_stats(Agent &agent,Agent &parent, const size_t &timeStep, std::ostringstream &str_stats) {
        str_stats << timeStep << "," << agent.get_ID() << "," << parent.get_ID() << "," << agent.get_skill(0) << "," << agent.get_skill(1)<< ","<< agent.get_skill_gen(0) << "," << agent.get_skill_gen(1);
//        for (size_t i = 0; i < N_PERCEPTIONS; i++) {
//            for (size_t j = 0; j < N_OUTPUTS; j++) {
//                str_stats << "," << agent.get_weights(i,j);
//            }
//        }
        str_stats << std::endl;
    }

    void Population::output_grid() {
        if (popID_ == 0) {
            size_t i(0), j(0);
            std::string fname_out("vis/Environment.vtk");
            std::ofstream fout;
            fout.open(fname_out.c_str(), std::ios::out);
            if (!fout) {
                std::cerr << "Failed to open " << fname_out << "\n";
                exit(1);
            }
            //std::cout<<" Writing grid to "<<fname_out<<"\n";

            fout << "# vtk DataFile Version 3.0\nvtk_output\nASCII\nDATASET RECTILINEAR_GRID\n";
            fout << "DIMENSIONS " << this->field_size_ + 1 << " " << this->field_size_ + 1 << " " << 1 << "\n";
            //Inserting X coordinates
            fout << "X_COORDINATES " << this->field_size_ + 1 << " float\n";
            for (i = 0; i < this->field_size_ + 1; ++i)
                fout << i << " ";
            //Inserting Y coordinates
            fout << "\nY_COORDINATES " << this->field_size_ + 1 << " float\n";
            for (i = 0; i < this->field_size_ + 1; ++i)
                fout << i << " ";
            fout << "\nZ_COORDINATES 1 float\n0\n";
            fout << "CELL_DATA " << this->field_size_ * this->field_size_ << "\n";
            fout << "FIELD FieldData 1\n";
            fout << "dummy_data 1 " << this->field_size_ * this->field_size_ << " integer\n";

            for (i = 0; i < this->field_size_; ++i) {
                for (j = 0; j < this->field_size_; ++j) {
                    fout << i * this->field_size_ + j << " ";
                }
            }
            fout.close();
            //std::cout << "\n#Finished writing grid" <<std::endl;
        }
    }

    void Population::output_food_fields(int tStep) {
        size_t i(0);
        float x, y;
        int cells_w_food(0), food(0);
        std::stringstream fname_out;
        fname_out << "vis/food_" << std::setw(5) << std::setfill('0') << popID_ << "_" << std::setw(5) << std::setfill('0') << tStep << ".vtk";
        std::ofstream fout;
        std::ostringstream position, value, type, lock;
        fout.open(fname_out.str(), std::ios::out);
        if (!fout) {
            std::cerr << "Failed to open " << fname_out.str() << "\n";
            exit(1);
        }
        for (i = 0; i < fields_.size(); i++) {
            for (int idx = 0; idx < FOOD_TYPES; idx++) {
                food = fields_[i].get_food(idx);
                if (food > 0)
                    cells_w_food++;
                // ---------- log information about fields ----------
                if (food > 0) {
                    x = (float) ((int) i % field_size_) + 0.25 + 0.5 * idx;
                    y = (float) ((int) i / field_size_) + 0.25 + 0.5 * idx;
                    position << x << " " << y << " 0\n";
                    value << food << "\n";
                    type << idx << "\n";
                    lock << (int) fields_[i].is_food_unlocked(idx) << "\n";
                }
            }
        }

        fout << "# vtk DataFile Version 3.1\nvtk_food_output\nASCII\nDATASET UNSTRUCTURED_GRID\n";
        fout << "POINTS " << cells_w_food << " FLOAT\n";
        fout << position.str() << "\n";
        fout << "POINT_DATA " << cells_w_food << "\n";
        fout << "SCALARS food_quantity float\nLOOKUP_TABLE default\n";
        fout << value.str();
        fout << "SCALARS food_type float\nLOOKUP_TABLE default\n";
        fout << type.str();
        fout << "SCALARS food_lock integer\nLOOKUP_TABLE default\n";
        fout << lock.str();
        fout.close();
        //std::cout << "\n#Finished writing food file" <<std::endl;
    }

    void Population::output_agents(int tStep) {
        size_t i(0);
        Agent* agent;
        float x(0), y(0);
        int field(0);
        std::stringstream fname_out;
        fname_out << "vis/agents_" << std::setw(5) << std::setfill('0') << popID_ << "_" << std::setw(5) << std::setfill('0') << tStep << ".vtk";
        std::ofstream fout;
        std::ostringstream position, id, energy, seed, skill;

        fout.open(fname_out.str(), std::ios::out);
        if (!fout) {
            std::cerr << "Failed to open " << fname_out.str() << "\n";
            exit(1);
        }

        for (i = 0; i < this->population_.size(); i++) {
            agent = &(population_[i].first);
            field = population_[i].second;
            x = (float) ((int) field % field_size_) + 0.5;
            y = (float) ((int) field / field_size_) + 0.5;
            position << x << " " << y << " 0\n";
            id << agent->get_ID() << "\n";
            energy << (float) agent->get_energy() << "\n";
            seed << agent->get_seed() << "\n";
            skill << agent->get_skill(1) << "\n";
        }

        fout << "# vtk DataFile Version 3.1\nvtk_food_output\nASCII\nDATASET UNSTRUCTURED_GRID\n";
        fout << "POINTS " << this->population_.size() << " FLOAT\n";
        fout << position.str() << "\n";
        fout << "POINT_DATA " << this->population_.size() << "\n";
        fout << "SCALARS sim_id integer\nLOOKUP_TABLE default\n";
        fout << id.str();
        fout << "SCALARS energy float\nLOOKUP_TABLE default\n";
        fout << energy.str();
        fout << "SCALARS seed integer\nLOOKUP_TABLE default\n";
        fout << seed.str();
        fout << "SCALARS skill float\nLOOKUP_TABLE default\n";
        fout << skill.str();
        fout.close();
        //std::cout << "\n#Finished writing agent file" <<std::endl;
    }

    void Population::save_population(std::string filename){
        //Saves the genome
        std::ofstream os(filename);
        cereal::XMLOutputArchive oarchive(os);
        oarchive(CEREAL_NVP(population_));
    }


    void Population::load_population(std::string filename){
        //Loads the genome
        std::ifstream is(filename);
        if(!is){
            std::cerr<<"Warning, file "<<filename<<" does not exist"<<std::endl;
        } else {
        cereal::XMLInputArchive iarchive(is); // Create an input archive
        iarchive(population_); // Read the data from the archive
    }
    }

}

//Needs to be checked if these operators do not poison the default namespace

//! Template classes for vector-vector and vector-scalar operations
template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}
template <typename T>
std::vector<T>& operator+=(std::vector<T>& a, const std::vector<T>& b)
{
    a=a+b;
    return a;
}
template <typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::minus<T>());
    return result;
}
template <typename T>
std::vector<T> operator*(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::multiplies<T>());
    return result;
}
template <typename T>
std::vector<T> operator/(const std::vector<T>& a, const T& b)
{

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), std::back_inserter(result),
                   std::bind2nd(std::divides<T>(),b));
    return result;
}


namespace Joleste {
    //! Nicely print statistics about the (sub)population
    std::string Population::print_pop(std::vector<population_type> pop) {
        std::ostringstream retval;
        std::cout<<"Testing population of "<<pop.size()<<" agents"<<std::endl;
        long int avg=0;
        for(auto i=pop.begin();i!=pop.end();i++) {
            std::cout<<"agent "<<get_agent_at(i).get_ID()<<" has energy "<<get_agent_at(i).get_energy()<<std::endl;
            avg+=get_agent_at(i).get_energy();
            retval<<get_agent_at(i).get_ID()<<","<<get_agent_at(i).get_energy()<<","<<get_agent_at(i).get_seed()<<std::endl;}
        std::cout<<"Population avg energy: "<<avg/(double)pop.size()<<std::endl;
        std::vector<std::vector<double> > stats = write_pop(pop);
        std::cout<<"Avg population performance= "<<std::endl;
        for(size_t i=0;i<stats[0].size();i++)
            std::cout<<stats[0][i]<<" ("<<std::sqrt(stats[1][i])<<") "<<stats[2][i]<<std::endl;
        std::cout<<std::endl<<"Values are: 0=north, 1=west, 2=east, 3=south, 4=eat"<<std::endl;;
        return retval.str();
    }

    //! Nicely print statistics about the subpopulation with given seed
    std::string Population::print_pop(int seed){
        std::vector<population_type> temp_pop;
        std::for_each(population_.begin(),population_.end(),[this,seed,&temp_pop](population_type a)
                 {
                     if(get_agent_at(a).get_seed()==seed)
                         temp_pop.push_back(a);
                 });
        return print_pop(temp_pop);
    }

    //! Return a vector with statistics about the (sub)population
    /*!
      Computes how close are the agents of the population to the "correct" (herding) behavior.
    */
    std::vector<std::vector<double>> Population::write_pop(std::vector<population_type> pop) {
        std::vector<std::vector <double> > result;
        int lng = N_PERCEPTIONS;
#ifndef INTERACT
        Genome::actions_type corrects={Agent::aton("eat"),
                                       Agent::aton("north"),
                                       Agent::aton("west"),
                                       Agent::aton("east"),
                                       Agent::aton("south")};
#elif defined INTERACT
#ifdef INVISIBLE_FOOD
        Genome::actions_type corrects={Agent::aton("eat"),
                                       Agent::aton("north"),
                                       Agent::aton("west"),
                                       Agent::aton("east"),
                                       Agent::aton("south")};
#elif !defined INVISIBLE_FOOD
        Genome::actions_type corrects={Agent::aton("eat"),
                                       Agent::aton("north"),
                                       Agent::aton("west"),
                                       Agent::aton("east"),
                                       Agent::aton("south"),
                                       Agent::aton("eat"),
                                       Agent::aton("north"),
                                       Agent::aton("west"),
                                       Agent::aton("east"),
                                       Agent::aton("south")};
#endif
#endif
        std::vector<double> sums(lng,0);
        std::vector<double> errors(lng,0);
        for(auto i=pop.begin();i!=pop.end();i++)
            sums+=get_agent_at(i).test_agent();
        sums=sums/(double)pop.size();
        for(auto i=pop.begin();i!=pop.end();i++)
            errors+=(sums-get_agent_at(i).test_agent())*(sums-get_agent_at(i).test_agent());
        errors=errors/(double)pop.size();

        result.push_back(sums);
        result.push_back(errors);
        result.push_back(corrects);
        return result;
    }

}
