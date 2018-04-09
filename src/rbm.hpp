/**
 * @file RBM.hpp
 * @brief RBM and DBN structures
 * @author: Leonel Aguilar - leonel.aguilar.m/at/gmail.com
 */
#ifndef RBM_DBN_HPP
#define RBM_DBN_HPP

#include <random>
#include <unordered_map>
#include <iostream> 
#include <iomanip>
#include <sstream>
#include <assert.h>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <cmath>
#include "cereal/cereal.hpp"


using Eigen::MatrixXd;
namespace RBM_DBN {

    class RBM {
    public:
        uint nfeatures; // Size of input layer
        uint hidSize; // Size of hidden layer
        double deNorm;
        MatrixXd b; // hidden layer bias
        MatrixXd c; // visible layer bias
        MatrixXd w; // weights

        double lrateW; //Learning rate for weights 
        double lrateC; //Learning rate for biases of visible units 
        double lrateB; //Learning rate for biases of hidden units 
        double weightcost;

        //Considering momentum (previous update)
        double initialmomentum;
        double finalmomentum;
        double momentum;
        MatrixXd incW;
        MatrixXd incB;
        MatrixXd incC;


    public:

        RBM(uint nfeatures_, uint hidSize_, double deNorm_);

        void lp1NormalizeCol(MatrixXd &m) const;

        void Init();
        void matRandomInit(MatrixXd &m, int rows, int cols, double scaler);
        MatrixXd getBernoulliMatrix(const MatrixXd &prob) const;
        std::vector<double> get_weights(std::vector<double> inputs)const;

        //Propagation operations
        MatrixXd forwAct(const MatrixXd &data, const int batchSize)const;
        MatrixXd forwProbAct(const MatrixXd &data, const int batchSize)const;
        MatrixXd backAct(const MatrixXd &HIDProbact, const int batchSize)const;
        MatrixXd backProbAct(const MatrixXd &HIDProbact, const int batchSize)const;
        MatrixXd backActFromProb(const MatrixXd &HIDProbact, const int batchSize)const;

        //Training
        MatrixXd train_(const MatrixXd &data, const uint epoches, int batchSize, int cd_k, bool mfinal);
        MatrixXd train2_(const MatrixXd &data, MatrixXd &posHIDprobs, const uint epoches, int batchSize, int cd_k, bool mfinal);
        MatrixXd train3_(const MatrixXd &data, MatrixXd &posHIDprobs, const uint epoches, int batchSize, int cd_k, bool mfinal);
        void train(std::vector<double> last_state, std::vector<double> des_current_qvalues, int epochs);

        //Helper functions
        static double sig_(double x) // the functor we want to apply
        {
            return 1.0 / (std::exp(-x) + 1);
        }

        MatrixXd sigmoid(const MatrixXd &M) const {
            return M.unaryExpr(&RBM::sig_);
        }

        static double pow2_(double x) // the functor we want to apply
        {
            return std::pow(x, 2);
        }

        MatrixXd
        pow2(const MatrixXd &M) {
            return M.unaryExpr(&pow2_);
        }
        
        template<class Archive>
        void serialize(Archive & archive) {
            std::cout << " SAVING NEEDS FIXING " << std::endl;
            archive(CEREAL_NVP(deNorm)); //,CEREAL_NVP(b),CEREAL_NVP(c),CEREAL_NVP(w)); // serialize things by passing them to the archive
        }

    };

    class DBN {
    public:
        std::vector<RBM> layers;
        int n_layers;

        DBN(int nfeatures_, int hidSize_, double deNorm_);

        void train(std::vector<double> last_state, std::vector<double> des_current_qvalues);

        std::vector<double> get_weights(std::vector<double> inputs);
        void Init();

        template<class Archive>
        void serialize(Archive & archive) {
            std::cout << " SAVING NEEDS FIXING " << std::endl;
            archive(CEREAL_NVP(layers[0])); //,CEREAL_NVP(b),CEREAL_NVP(c),CEREAL_NVP(w)); // serialize things by passing them to the archive
        }
    };

}
#endif
