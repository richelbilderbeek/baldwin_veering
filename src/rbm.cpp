/**
 * @file rbm.cpp
 * @brief RBM and DBN structures
 * @author: Leonel Aguilar - leonel.aguilar.m/at/gmail.com
 */

#include "./rbm.hpp"
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
#include "constants.h"
#include <limits>


using Eigen::MatrixXd;
namespace RBM_DBN {

    RBM::RBM(uint nfeatures_, uint hidSize_, double deNorm_) : nfeatures(nfeatures_), hidSize(hidSize_ + 1), deNorm(deNorm_) {
        Init();
        lrateW = 0.01; //Learning rate for weights
        lrateC = 0.01; //Learning rate for biases of visible units
        lrateB = 0.01; //Learning rate for biases of hidden units
        weightcost = 0.0002;

        //std::cout<<" Nfeatures "<<nfeatures_<<" hid size "<<hidSize_<<std::endl;

        initialmomentum = 0.5;
        finalmomentum = 0.9;

        incW = MatrixXd::Constant(w.rows(), w.cols(), 0.0);
        incB = MatrixXd::Constant(b.rows(), b.cols(), 0.0);
        incC = MatrixXd::Constant(c.rows(), c.cols(), 0.0);
    }

    void RBM::lp1NormalizeCol(MatrixXd &m) const {
        MatrixXd norms = m.colwise().lpNorm<1>();
        for (uint i = 0; i < m.rows(); i++) {
            for (uint j = 0; j < m.cols(); j++) {
                if (norms(0, j))
                    m(i, j) = m(i, j) / norms(0, j);
            }
        }
    }

    void RBM::Init() {
        matRandomInit(w, nfeatures, hidSize, 0.12);
        matRandomInit(b, hidSize, 1, 0); //Bias hidden units
        matRandomInit(c, nfeatures, 1, 0); //Bias visible units
        incW = MatrixXd::Constant(w.rows(), w.cols(), 0.0);
        incB = MatrixXd::Constant(b.rows(), b.cols(), 0.0);
        incC = MatrixXd::Constant(c.rows(), c.cols(), 0.0);
    }

    void RBM::matRandomInit(MatrixXd &m, int rows, int cols, double scaler) {
        m = (MatrixXd::Random(rows, cols)) * scaler;
    }

    MatrixXd RBM::getBernoulliMatrix(const MatrixXd &prob) const {
        MatrixXd ran = (MatrixXd::Constant(prob.rows(), prob.cols(), 1) + MatrixXd::Random(prob.rows(), prob.cols()))*0.5;
        MatrixXd res = MatrixXd::Zero(prob.rows(), prob.cols());
        for (uint i = 0; i < prob.rows(); i++) {
            for (uint j = 0; j < prob.cols(); j++) {
                if (prob(i, j) > ran(i, j)) {
                    res(i, j) = 1;
                }
            }
        }
        return res;
    }

    std::vector<double> RBM::get_weights(std::vector<double> inputs)const {
        assert(inputs.size() == nfeatures);
        std::vector<double> vresults(hidSize - 1);
        MatrixXd data = MatrixXd::Constant(nfeatures, 1, 1.0);
        for (uint i = 0; i < nfeatures; i++) {
            data(i, 0) = inputs[i];
        }

        MatrixXd result = forwAct(data, 1);
        double SdeNorm = result(0, 0);

        if (SdeNorm>std::numeric_limits<float>::epsilon()){
            SdeNorm = (deNorm / SdeNorm);
        }

        for (uint i = 0; i < (hidSize - 1); i++) {
            if(result(i + 1, 0)> 100*std::numeric_limits<float>::epsilon()){
                vresults[i] = std::pow((result(i + 1, 0) * SdeNorm)-0.1, 4.0);
            }else{
                vresults[i] = 0.0;
            }
            //No fraction bias
            //vresults[i]=(result(i+1,0))*deNorm-1;
            if (vresults[i] < 0)vresults[i] = 0.0;
            if(vresults[i]>2.0*MAX_REWARD) vresults[i]=2.0*MAX_REWARD;
        }
        return vresults;

    }

    MatrixXd RBM::forwAct(const MatrixXd &data, const int batchSize)const {
        MatrixXd HIDprobs;
        MatrixXd HIDact;

        HIDprobs = sigmoid(w.transpose() * data + b.replicate(1, batchSize));
        lp1NormalizeCol(HIDprobs);
        HIDact = HIDprobs.rowwise().sum() / batchSize;
        //std::cout<<"\n Probs "<<HIDprobs<<std::endl;
        return HIDact;
    }

    MatrixXd RBM::forwProbAct(const MatrixXd &data, const int batchSize)const {
        MatrixXd HIDprobs;

        HIDprobs = sigmoid(w.transpose() * data + b.replicate(1, batchSize));
        lp1NormalizeCol(HIDprobs);
        return HIDprobs;
    }

    MatrixXd RBM::backAct(const MatrixXd &HIDProbact, const int batchSize)const {
        MatrixXd VISprobs;
        MatrixXd VISact;

        VISprobs = sigmoid(w * HIDProbact + c.replicate(1, batchSize));
        lp1NormalizeCol(VISprobs);
        VISact = VISprobs.rowwise().sum() / batchSize;
        return VISact;
    }

    MatrixXd RBM::backProbAct(const MatrixXd &HIDProbact, const int batchSize)const {
        MatrixXd VISprobs;
        MatrixXd VISact;

        VISprobs = sigmoid(w * HIDProbact + c.replicate(1, batchSize));
        lp1NormalizeCol(VISprobs);
        return VISprobs;
    }

    MatrixXd RBM::backActFromProb(const MatrixXd &HIDProbact, const int batchSize)const {
        MatrixXd VISprobs;
        MatrixXd VISact;
        MatrixXd HIDstates;
        MatrixXd negdata;
        HIDstates = getBernoulliMatrix(HIDProbact);
        negdata = sigmoid(w * HIDstates + c.replicate(1, batchSize));
        negdata = getBernoulliMatrix(negdata);
        return negdata;
    }

    MatrixXd RBM::train_(const MatrixXd &data, const uint epoches, int batchSize, int cd_k, bool mfinal) {

        //Positive phase
        MatrixXd posHIDprobs;
        MatrixXd posHIDact;
        MatrixXd posHIDprobs_temp;
        MatrixXd posHIDstates;
        MatrixXd posprods;
        MatrixXd posVISact;

        //Negative phase
        MatrixXd negHIDprobs;
        MatrixXd negHIDact;
        MatrixXd negVISact;
        MatrixXd negprods;
        MatrixXd negdata;

        for (uint ep = 0; ep < epoches; ep++) {

            if ((float) ep / (float) epoches > 0.5)mfinal = true;
            // start positive phase
            //data2 = getBernoulliMatrix2(data2);//CHECK
            posHIDprobs = sigmoid(w.transpose() * data + b.replicate(1, batchSize));

            lp1NormalizeCol(posHIDprobs);

            posprods = data * posHIDprobs.transpose() / batchSize;
            posHIDact = posHIDprobs.rowwise().sum() / batchSize;

            posVISact = data.rowwise().sum() / batchSize;

            // end of positive phase
            posHIDprobs_temp = posHIDprobs;


            // start negative phase
            // CD-K alg
            for (int i = 0; i < cd_k; i++) {

                posHIDstates = getBernoulliMatrix(posHIDprobs_temp);
                negdata = sigmoid(w * posHIDstates + c.replicate(1, batchSize));
                negdata = getBernoulliMatrix(negdata);
                posHIDprobs_temp = sigmoid(w.transpose() * negdata + b.replicate(1, batchSize));
                lp1NormalizeCol(posHIDprobs_temp);
            }
            negHIDprobs = posHIDprobs_temp;
            negprods = negdata * negHIDprobs.transpose() / batchSize;
            negHIDact = negHIDprobs.rowwise().sum() / batchSize;
            negVISact = negdata.rowwise().sum() / batchSize;

            //end of negative phase
            if (mfinal) momentum = finalmomentum;
            else momentum = initialmomentum;

            // update weights and biases
            incW = momentum * incW + lrateW * ((posprods - negprods) - weightcost * w);
            incC = momentum * incC + lrateC * (posVISact - negVISact);
            incB = momentum * incB + lrateB * (posHIDact - negHIDact);
            w += incW;
            c += incC;
            b += incB;

        }

        return negdata;
    }

    MatrixXd RBM::train2_(const MatrixXd &data, MatrixXd &posHIDprobs, const uint epoches, int batchSize, int cd_k, bool mfinal) {

        //Positive phase
        //MatrixXd posHIDprobs;
        MatrixXd posHIDact;
        MatrixXd posHIDprobs_temp;
        MatrixXd posHIDstates;
        MatrixXd posprods;
        MatrixXd posVISact;

        //Negative phase
        MatrixXd negHIDprobs;
        MatrixXd negHIDact;
        MatrixXd negVISact;
        MatrixXd negprods;
        MatrixXd negdata;

        for (uint ep = 0; ep < epoches; ep++) {

            if ((float) ep / (float) epoches > 0.5)mfinal = true;
            // start positive phase
            posprods = data * posHIDprobs.transpose() / batchSize;
            posHIDact = posHIDprobs.rowwise().sum() / batchSize;

            posVISact = data.rowwise().sum() / batchSize;

            // end of positive phase
            posHIDprobs_temp = posHIDprobs;


            // start negative phase
            // CD-K alg
            for (int i = 0; i < cd_k; i++) {

                posHIDstates = getBernoulliMatrix(posHIDprobs_temp);
                negdata = sigmoid(w * posHIDstates + c.replicate(1, batchSize));
                negdata = getBernoulliMatrix(negdata);
                posHIDprobs_temp = sigmoid(w.transpose() * negdata + b.replicate(1, batchSize));
                lp1NormalizeCol(posHIDprobs_temp);
            }
            negHIDprobs = posHIDprobs_temp;
            negprods = negdata * negHIDprobs.transpose() / batchSize;
            negHIDact = negHIDprobs.rowwise().sum() / batchSize;
            negVISact = negdata.rowwise().sum() / batchSize;

            //end of negative phase
            if (mfinal) momentum = finalmomentum;
            else momentum = initialmomentum;

            // update weights and biases
            incW = momentum * incW + lrateW * ((posprods - negprods) - weightcost * w);
            incC = momentum * incC + lrateC * (posVISact - negVISact);
            incB = momentum * incB + lrateB * (posHIDact - negHIDact);
            w += incW;
            c += incC;
            b += incB;

        }

        return negdata;
    }

    MatrixXd RBM::train3_(const MatrixXd &data, MatrixXd &posHIDprobs, const uint epoches, int batchSize, int cd_k, bool mfinal) {

        //Positive phase
        //MatrixXd posHIDprobs;
        MatrixXd posHIDact;
        MatrixXd posHIDprobs_temp;
        MatrixXd posHIDstates;
        MatrixXd posprods;
        MatrixXd posVISact;

        //Negative phase
        MatrixXd negHIDprobs;
        MatrixXd negHIDact;
        MatrixXd negVISact;
        MatrixXd negprods;
        MatrixXd negdata;

        for (uint ep = 0; ep < epoches; ep++) {

            if ((float) ep / (float) epoches > 0.5)mfinal = true;
            // start positive phase
            //v*ht
            posprods = data * posHIDprobs.transpose() / batchSize;
            posHIDact = posHIDprobs.rowwise().sum() / batchSize;

            posVISact = data.rowwise().sum() / batchSize;

            // end of positive phase
            posHIDprobs_temp = posHIDprobs;

            negdata = data;
            negHIDprobs = this->forwProbAct(negdata, batchSize);

            negprods = negdata * negHIDprobs.transpose() / batchSize;
            negHIDact = negHIDprobs.rowwise().sum() / batchSize;
            negVISact = negdata.rowwise().sum() / batchSize;

            //end of negative phase
            if (mfinal) momentum = finalmomentum;
            else momentum = initialmomentum;

            // update weights and biases
            incW = momentum * incW + lrateW * ((posprods - negprods) - weightcost * w);
            incC = momentum * incC + lrateC * (posVISact - negVISact);
            incB = momentum * incB + lrateB * (posHIDact - negHIDact);
            w += incW;
            c += incC;
            b += incB;

        }

        return negdata;
    }

    void RBM::train(std::vector<double> last_state, std::vector<double> des_current_qvalues, int epochs) {
        assert(last_state.size() == nfeatures);
        assert(des_current_qvalues.size() + 1 == hidSize);
        std::vector<double> vresults(hidSize);
        MatrixXd data = MatrixXd::Constant(nfeatures, 1, 1.0);
        MatrixXd HIDprobs = MatrixXd::Constant(hidSize, 1, 1.0);
        for (uint i = 0; i < nfeatures; i++) {
            data(i, 0) = last_state[i];
        }

        double filler(deNorm);
        //double Hsum(0);
        for (uint i = 1; i < hidSize; i++) {
            //Currently no negative reward is allowed
            if(des_current_qvalues[i - 1]<0) des_current_qvalues[i - 1]=0;
            if(des_current_qvalues[i - 1]>2.0*MAX_REWARD) des_current_qvalues[i - 1]=2.0*MAX_REWARD;
            if(des_current_qvalues[i - 1]<10*std::numeric_limits<float>::epsilon()){
                HIDprobs(i, 0) = 0.0+0.1;
            }else{
                HIDprobs(i, 0) = std::pow(des_current_qvalues[i - 1], 1. / 4.)+0.1;
            }
        }
        filler = deNorm;
        assert(filler >= 0);
        HIDprobs(0, 0) = filler;
        lp1NormalizeCol(HIDprobs);
        train3_(data, HIDprobs, epochs, 1, 1, false);
    }

    DBN::DBN(int nfeatures_, int hidSize_, double deNorm_) {
        n_layers = 2;
        layers.push_back(RBM(nfeatures_, nfeatures_ - 1, deNorm_));
        layers.push_back(RBM(nfeatures_, hidSize_, deNorm_));
    }

    void DBN::train(std::vector<double> last_state, std::vector<double> des_current_qvalues) {

        assert(last_state.size() == layers[0].nfeatures);
        assert(des_current_qvalues.size() + 1 == layers[n_layers - 1].hidSize);
        MatrixXd data = MatrixXd::Constant(layers[0].nfeatures, 1, 1.0);

        for (uint i = 0; i < layers[0].nfeatures; i++) {
            data(i, 0) = last_state[i];
        }

        MatrixXd data1, negdata0, negdata1;
        negdata0 = layers[0].train_(data, 1000, 1, 1, false);
        data1 = layers[0].forwAct(data, 1);
        data1 = layers[0].getBernoulliMatrix(data1);
        std::vector<double> vdata1(layers[1].nfeatures, -1);
        for (uint i = 0; i < layers[1].nfeatures; i++) {
            vdata1[i] = data1(i, 0);
        }
        layers[1].train(vdata1, des_current_qvalues, 1);
        //Check
        std::vector<double> hidAct = layers[1].get_weights(vdata1);
        std::cout << " TRAINED \n";
        for (auto a : hidAct) std::cout << a << " , ";
        std::cout << std::endl;
        std::cout << " DESIRED \n";
        for (auto a : des_current_qvalues) std::cout << a << " , ";
        std::cout << std::endl;
    }

    std::vector<double> DBN::get_weights(std::vector<double> inputs) {
        assert(inputs.size() == layers[0].nfeatures);
        std::vector<double> vresults;
        std::vector<double> vdata1(layers[1].nfeatures, -1);
        MatrixXd data = MatrixXd::Constant(layers[0].nfeatures, 1, 1.0);
        for (uint i = 0; i < layers[0].nfeatures; i++) {
            data(i, 0) = inputs[i];
        }
        MatrixXd data1;
        data1 = layers[0].forwAct(data, 1);
        data1 = layers[0].getBernoulliMatrix(data1);
        for (uint i = 0; i < layers[1].nfeatures; i++) {
            vdata1[i] = data1(i, 0);
        }
        vresults = layers[1].get_weights(vdata1);
        return vresults;
    }

    void DBN::Init() {
        layers[0].Init();
        layers[1].Init();
    }
}
