/**
 * @file ann.hpp
 * @brief wrapper function to the CNN library. Currently provides only basic
 * functionality
 * @author: Leonel Aguilar - leonel.aguilar.m/at/gmail.com
 */

#ifndef ANN_HPP
#define ANN_HPP

#include "constants.h"
#include "tiny-dnn/tiny_dnn/tiny_dnn.h"
#include <sstream>      // std::stringstream

namespace ANN
{
    using namespace tiny_dnn;
    using namespace tiny_dnn::activation;
    using namespace tiny_dnn::layers;
    class NeuralNetwork
    {
    public:
        network<sequential> net;
        gradient_descent opt;
        size_t in_dimension;
        size_t out_dimension;
    public:
        NeuralNetwork(size_t in_dimension_,size_t out_dimension_):
        net("agent_brain"),
        in_dimension(in_dimension_),
        out_dimension(out_dimension_)
        {
            // Define layers
            net << fully_connected_layer<relu>(in_dimension,in_dimension*5) // 5x1x10
                << fully_connected_layer<relu>(in_dimension*5, out_dimension*5) //40x3 (left, right, eat)
                << fully_connected_layer<identity>(out_dimension*5, out_dimension); //40x3 (left, right, eat)

            std::vector<vec_t> dummy2;
            std::vector<vec_t> dummy_qvals2;
            std::vector<tensor_t> dummy2_tensor;
            std::vector<tensor_t> dummy_qvals2_tensor;
            normalize_tensor(dummy2,dummy2_tensor);
            normalize_tensor(dummy_qvals2,dummy_qvals2_tensor);
            net.fit_INIT<mse>(opt,dummy2_tensor,dummy_qvals2_tensor,1,1,nop,nop,false,1);

            //Check if dimensions match
            assert(net.in_data_size() == in_dimension);
            assert(net.out_data_size() == out_dimension);
        }
    //Replicate constructor
    NeuralNetwork(const NeuralNetwork &parent_nn,const int &ID):
        //net(parent_nn.net),
        opt(parent_nn.opt),
        in_dimension(parent_nn.in_dimension),
        out_dimension(parent_nn.out_dimension)
        {
            net << fully_connected_layer<relu>(in_dimension,in_dimension*5) // 5x1x10
                << fully_connected_layer<relu>(in_dimension*5, out_dimension*5) //40x3 (left, right, eat)
                << fully_connected_layer<identity>(out_dimension*5, out_dimension); //40x3 (left, right, eat)
            std::stringstream is;
            is<<parent_nn.net;
            net.load(is);

            std::vector<vec_t> dummy2;
            std::vector<vec_t> dummy_qvals2;
            std::vector<tensor_t> dummy2_tensor;
            std::vector<tensor_t> dummy_qvals2_tensor;
            normalize_tensor(dummy2,dummy2_tensor);
            normalize_tensor(dummy_qvals2,dummy_qvals2_tensor);
            net.fit_INIT<mse>(opt,dummy2_tensor,dummy_qvals2_tensor,1,1,nop,nop,false,1);

            //Check if dimensions match
            assert(net.in_data_size() == in_dimension);
            assert(net.out_data_size() == out_dimension);
        }

        void train_single(const std::vector<double> &state,const std::vector<double> &q_values,int epoches=50){
            train_single_(net,state,q_values,epoches);
        }
        void train_single_(network<sequential> &ann_net,const std::vector<double> &state,const std::vector<double> &q_values,int epoches=50){
            vec_t state_temp;
            vec_t q_values_temp;
            wrap_type(state,state_temp,in_dimension);
            wrap_type(q_values,q_values_temp,out_dimension);

            std::vector<vec_t> state_temp_dummy;
            std::vector<vec_t> q_values_temp_dummy;
            for(int i=0;i<1;i++){
                state_temp_dummy.push_back(state_temp);
                q_values_temp_dummy.push_back(q_values_temp);
            }
            std::vector<tensor_t> current_state_tensor;
            std::vector<tensor_t> q_values_C_tensor;
            normalize_tensor(state_temp_dummy,current_state_tensor);
            normalize_tensor(q_values_temp_dummy,q_values_C_tensor);
            ann_net.fit_WRAP<mse>(opt,current_state_tensor,q_values_C_tensor,1,epoches,nop,nop,false,1);
        }

        void train_batch(const std::vector<std::vector<double> > &states,const std::vector<std::vector<double> > &q_values,int epoches=50,int batch_size=0){
            if(batch_size==0) batch_size=states.size();
            assert(states.size()==q_values.size());
            vec_t state_temp;
            vec_t q_values_temp;
            std::vector<vec_t> state_temp_dummy;
            std::vector<vec_t> q_values_temp_dummy;
            for(uint i=0;i<states.size();i++){
                wrap_type(states[i],state_temp,in_dimension);
                wrap_type(q_values[i],q_values_temp,out_dimension);
                state_temp_dummy.push_back(state_temp);
                q_values_temp_dummy.push_back(q_values_temp);
            }
            std::vector<tensor_t> current_state_tensor;
            std::vector<tensor_t> q_values_C_tensor;
            normalize_tensor(state_temp_dummy,current_state_tensor);
            normalize_tensor(q_values_temp_dummy,q_values_C_tensor);
            if(!net.fit_WRAP<mse>(opt,current_state_tensor,q_values_C_tensor,batch_size,epoches,nop,nop,false,1)){
                std::cerr<<" TRAIN BATCH FAILED "<<std::endl;
            }
        }

        label_t get_action(std::vector<double> &state){
            vec_t state_temp;
            wrap_type(state,state_temp,in_dimension);
            return net.predict_label(state_temp);
        }
        std::vector<double> get_QValues(const std::vector<double> &state,bool clip=true)const{
            vec_t state_temp, out_temp;
            std::vector<double> out;
            wrap_type(state,state_temp,in_dimension);
            //TODO: NEED TO MAKE PREDICT CONST
            network<sequential>& nc_net = const_cast< network<sequential>&>(net);
            out_temp=nc_net.predict(state_temp);
            wrap_type(out_temp,out,out_dimension);
            if(clip){
                for(auto &a:out){
                    if(a!=a){
                        std::cerr<<"\nError: DNN with nan weights\n ..exiting"<<std::endl;
                        abort();
                    }
                    if(a<(double)-MAX_REWARD_DRL*GAMMA_MULT)a=(double)-MAX_REWARD_DRL*GAMMA_MULT;
                    if(a>(double)MAX_REWARD_DRL*GAMMA_MULT)a=(double)MAX_REWARD_DRL*GAMMA_MULT;
                }
            }
            return out;
        }

        void reset(){
            int max_trials(100);
            bool success(false);
            int correct(0);

            double min_val(0.1);
            std::cout<<"Calling reset"<<std::endl;
            net.init_weight();
            std::vector<double> temp_qvalues_ini(out_dimension,min_val);
            std::vector<double> input_state_ini(in_dimension,1);
            train_single_(net,input_state_ini,temp_qvalues_ini,10);
           for(int i=0;i<max_trials;i++){
                    //Check if the init was successful
                    std::vector<double> temp_qvalues;
                    std::vector<double> input_state(in_dimension,1);
                    temp_qvalues=get_QValues(input_state,false);
                        std::cout<<" GOT "<<std::endl;
                        for(auto a:temp_qvalues){
                            std::cout<<a;
                        }
                        std::cout<<std::endl;

                    correct=0;
                    for(auto &v:temp_qvalues){
                        if((v<=min_val)&&(v>0)){
                            correct++;
                        }else{
                            v=min_val;
                        }
                    }

                    if((uint) correct == temp_qvalues.size()){
                        std::cout<<" CORRECT "<<std::endl;
                        for(auto a:temp_qvalues){
                            std::cout<<a;
                        }
                        std::cout<<std::endl;
                        success=true;
                        break;
                    }else{
                        std::cout<<" DESIRED "<<std::endl;
                        for(auto a:temp_qvalues){
                            std::cout<<a;
                        }
                        std::cout<<std::endl;
                        train_single_(net,input_state,temp_qvalues,1);
                        temp_qvalues=get_QValues(input_state);
                        std::cout<<" TRAINED "<<std::endl;
                        for(auto a:temp_qvalues){
                            std::cout<<a;
                        }
                        std::cout<<std::endl;
                    }
                }
                if(!success){
                    std::cerr<<"\nError: could not initialize DNN\n ..exiting"<<std::endl;
    //                abort();
                }
        }

        void Initialize(){

            std::vector<vec_t> dummy2;
            std::vector<vec_t> dummy_qvals2;
            std::vector<tensor_t> dummy2_tensor;
            std::vector<tensor_t> dummy_qvals2_tensor;
            normalize_tensor(dummy2,dummy2_tensor);
            normalize_tensor(dummy_qvals2,dummy_qvals2_tensor);
            net.fit_INIT<mse>(opt,dummy2_tensor,dummy_qvals2_tensor,1,1,nop,nop,false,1);
            network<sequential> net_temp;
            int max_trials(10);
            double terror_max(0.1);
            double terror(0);
            for(int i=0;i<max_trials;i++){
                net.init_weight();
                net_temp=net;
                std::vector<double> temp_qvalues(out_dimension,0);
                std::vector<double> input_state(in_dimension,1);
                temp_qvalues[0]=10;
                train_single_(net_temp,input_state,temp_qvalues,50);

                vec_t state_temp, out_temp;
                std::vector<double> out;
                wrap_type(input_state,state_temp,in_dimension);
                out_temp=net_temp.predict(state_temp);
                terror=0;
                for(uint j=0;j<out_dimension;j++){
                    terror+=std::fabs(out_temp[j]-temp_qvalues[j]);
                }
                if(terror<terror_max){
                    break;
                }

            }
        }

        template<typename T1,typename T2>
        void wrap_type(T1 &input, T2 &output, int vect_size) const {
            serial_size_t outdim = vect_size;
            assert(vect_size > 0);
            assert(outdim > 0);
            assert(input.size()==outdim);
            output.resize(vect_size);
            for(int i = 0; i < vect_size; i++){
                output[i]=input[i];
            }
        }

        void normalize_tensor(const std::vector<vec_t>& inputs, std::vector<tensor_t>& normalized) {
        normalized.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
            normalized.emplace_back(tensor_t{ inputs[i] });
        }

        void save_weights(std::string fname){
            // save
            std::ofstream ofs(fname.c_str());
            ofs << net;
        }

        void load_weights(std::string fname){
            // load
            std::ifstream ifs(fname.c_str());
            ifs >> net;
        }

        template<class Archive>
        void save(Archive & archive) const
        {
          net.to_archive(archive,content_type::weights_and_model);
        }

        template<class Archive>
        void load(Archive & archive)
        {
          net.from_archive(archive,content_type::weights_and_model);
        }
    };
} // end namespace ANN

#endif
