/**
 * CMA-ES, Covariance Matrix Adaptation Evolution Strategy
 * Copyright (c) 2014 Inria
 * Author: Emmanuel Benazera <emmanuel.benazera@lri.fr>
 *
 * This file is part of libcmaes.
 *
 * libcmaes is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libcmaes is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libcmaes.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ESOSTRATEGY_H
#define ESOSTRATEGY_H

#include "eo_matrix.h" // to include Eigen everywhere.
#include "candidate.h"
#include "eigenmvn.h"
#include "libcmaes_config.h"
#include "cmaparameters.h" // in order to pre-instanciate template into library.
#include "cmasolutions.h"
#include "llogging.h"

#include <iostream>
#include <random>

namespace libcmaes
{
    //typedef std::function<double (const double*, const int &n)> FitFunc;
    typedef std::function<dVec (const double*, const int &n)> GradFunc;

    typedef std::function<void(const dMat&, const dMat&)> EvalFunc;
    typedef std::function<dMat(void)> AskFunc;
    typedef std::function<void(void)> TellFunc;
  
    template <typename T> 
    int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }

    /**
     * \brief Main class describing an evolutionary optimization strategy.
     *        Every algorithm in libcmaes descends from this class, and bring
     *        its functionalities to an ESOptimizer object.
     */
    template<class TParameters,class TSolutions>
    class ESOStrategy
    {
    public:
        /**
        * \brief dummy constructor.
        */
        ESOStrategy(){};
    
        /**
         * \brief constructor
         * @param parameters optimization parameters
         */
        ESOStrategy(TParameters &parameters):_nevals(0),_niter(0),_parameters(parameters){
      
            _solutions = TSolutions(_parameters);
            if (parameters._uh){
        	    std::random_device rd;
                _uhgen = std::mt19937(rd());
                _uhgen.seed(static_cast<uint64_t>(time(nullptr)));
                _uhunif = std::uniform_real_distribution<>(0,1);
            }
        };

        /**
         * \brief constructor for starting from an existing solution.
         * @param parameters stochastic search parameters
         * @param solution solution object to start from
         */
        ESOStrategy(TParameters &parameters, const TSolutions &solutions):_nevals(0),_niter(0),_parameters(parameters){
            start_from_solution(solutions);
            if (parameters._uh){
                    std::random_device rd;
                _uhgen = std::mt19937(rd());
                _uhgen.seed(static_cast<uint64_t>(time(nullptr)));
                _uhunif = std::uniform_real_distribution<>(0,1);
            }
        };
        
    protected:
        ~ESOStrategy(){};

    public:
        /**
         * \brief Generates a set of candidate points.
         * @return A matrix whose rows contain the candidate points.
         */
        dMat ask();

        /**
         * \brief Updates the state of the stochastic search, and prepares
         *        for the next iteration.
         */
        void tell();
        
        /**
         * \brief increment iteration count.
         */
        void inc_iter(){
            _niter++;
            this->_solutions._niter++;
        };

        /**
         * \brief updates the consumed budget of objective function evaluations.
         * @param evals increment to the current consumed budget
         */
        void update_fevals(const int &evals){
            _nevals += evals;
            this->_solutions._nevals += evals;
        };

        /**
         * \brief sets the gradient function, if available.
         * @param gfunc gradient function
         */
        void set_gradient_func(GradFunc &gfunc) { _gfunc = gfunc; }
    
        /**
         * \brief starts optimization from a given solution object.
         * @param sol the solution object to start search from.
         */
        void start_from_solution(const TSolutions &sol){
            _parameters.set_x0(sol.best_candidate().get_x_dvec());
            this->_solutions = sol;
            this->_solutions.reset();
        }

        /**
         * \brief returns reference to current solution object
         * @return current solution object
         */
        TSolutions& get_solutions() { return this->_solutions; }

        /**
         * \brief returns reference to current optimization parameters object
         * @return current optimization parameters object
         */
        TParameters& get_parameters() { return _parameters; }

        /**
         * \brief uncertainty handling scheme that computes and uncertainty
         *        level based on a dual candidate ranking.
         */
        void uncertainty_handling(){
            std::sort(this->_solutions._candidates_uh.begin(),
                this->_solutions._candidates_uh.end(),
                [](const RankedCandidate &c1, const RankedCandidate &c2)
                { 
                bool lower = c1.get_fvalue() < c2.get_fvalue();
                return lower;
                });
            int pos = 0;
            auto vit = this->_solutions._candidates_uh.begin();
            while(vit!=this->_solutions._candidates_uh.end())
            {
            (*vit)._r1 = pos;
            ++vit;
            ++pos;
            }
            
            // sort second uh set of candidates
            std::sort(this->_solutions._candidates_uh.begin(),
                this->_solutions._candidates_uh.end(),
                [](const RankedCandidate &c1, const RankedCandidate &c2)
                { 
                bool lower = c1._fvalue_mut < c2._fvalue_mut;
                return lower;
                });
            pos = 0;
            vit = this->_solutions._candidates_uh.begin();
            while(vit!=this->_solutions._candidates_uh.end())
            {
            (*vit)._r2 = pos;
            ++vit;
            ++pos;
            }
            
            // compute delta
            vit = this->_solutions._candidates_uh.begin();
            while(vit!=this->_solutions._candidates_uh.end())
            {
            if ((*vit)._idx >= this->_solutions._lambda_reev)
            {
                ++vit;
                continue;
            }
            int diffr = (*vit)._r2 - (*vit)._r1;
            (*vit)._delta = diffr - sgn(diffr);
            ++vit;
            }
            double meandelta = std::accumulate(this->_solutions._candidates_uh.begin(),
                            this->_solutions._candidates_uh.end(),
                            0.0,
                            [](double sum, const RankedCandidate &c){ return sum + fabs(c._delta); });
            meandelta /= this->_solutions._lambda_reev;
            
            // compute uncertainty level
            double s = 0.0;
            for (size_t i=0;i<this->_solutions._candidates_uh.size();i++)
            {
            RankedCandidate rc = this->_solutions._candidates_uh.at(i);
            if (rc._idx >= this->_solutions._lambda_reev)
            continue;
            s += 2*fabs(rc._delta);
            double d1 = rc._r2 - static_cast<int>(rc._r2 > rc._r1);
            std::vector<double> dv;
            double fact = _parameters._thetauh*0.5;
            for (int j=1;j<2*_parameters._lambda;j++)
            dv.push_back(fabs(j-d1));
            std::nth_element(dv.begin(),dv.begin()+int(dv.size()*fact),dv.end());
            double comp1 = *(dv.begin()+int(dv.size()*fact));
            s -= comp1;
            
            double d2 = rc._r1 - static_cast<int>(rc._r1 > rc._r2);
            dv.clear();
            for (int j=1;j<2*_parameters._lambda;j++)
            dv.push_back(fabs(j-d2));
            std::nth_element(dv.begin(),dv.begin()+int(dv.size()*fact),dv.end());
            double comp2 = *(dv.begin()+int(dv.size()*fact));
            s -= comp2;
            }
            s /= static_cast<double>(this->_solutions._lambda_reev);
            this->_solutions._suh = s;
            
            // rerank according to r1 + r2
            int lreev = this->_solutions._lambda_reev;
            std::sort(this->_solutions._candidates_uh.begin(),
                this->_solutions._candidates_uh.end(),
                [lreev,meandelta](RankedCandidate const &c1, RankedCandidate const &c2)
                { 
                int s1 = c1._r1 + c1._r2;
                int s2 = c2._r2 + c2._r2;
                if (s1 == s2)
                {
                    if (c1._delta == c2._delta)
                    return c1.get_fvalue() + c1._fvalue_mut < c2.get_fvalue() + c2._fvalue_mut;
                    else
                    {
                    double c1d = c1._idx < lreev ? fabs(c1._delta) : meandelta;
                    double c2d = c2._idx < lreev ? fabs(c2._delta) : meandelta;
                    return c1d < c2d;
                    }
                }
                else return c1._r1 + c1._r2 < c2._r1 + c2._r2;
                });
            std::vector<Candidate> ncandidates;
            vit = this->_solutions._candidates_uh.begin();
            while(vit!=this->_solutions._candidates_uh.end())
            {
            ncandidates.push_back(this->_solutions._candidates.at((*vit)._idx));
            ++vit;
            }
            this->_solutions._candidates = ncandidates;
        };

        /**
         * \brief updates the two-point adaptation average rank difference for
         *        the step-size adaptation mechanism
         */
        void tpa_update(){
            int r1 = -1;
            int r2 = -1;
            for (size_t i=0;i<this->_solutions._candidates.size();i++)
            {
            if (r1 == -1 && this->_solutions._candidates.at(i).get_id() == this->_solutions._tpa_p1)
            {
                r1 = i;
            }
            if (r2 == -1 && this->_solutions._candidates.at(i).get_id() == this->_solutions._tpa_p2)
            {
                r2 = i;
            }
            if (r1 != -1 && r2 != -1)
            {
                break;
            }
            }
            int rank_diff = r2-r1;
            this->_solutions._tpa_s = (1.0 - _parameters._tpa_csigma) * this->_solutions._tpa_s
            + _parameters._tpa_csigma * rank_diff / (_parameters._lambda - 1.0);
        };

        // deprecated.
        Candidate best_solution() const{
            return this->_solutions.best_candidate();
        };

        void set_initial_elitist(const bool &e) { _initial_elitist = e; }
    
    protected:
        int _nevals;  /**< number of function evaluations. */
        int _niter;  /**< number of iterations. */
        TSolutions _solutions; /**< holder of the current set of solutions and the dynamic elemenst of the search state in general. */
        dMat _currentCandidates; /**< holder of the new set of solutions and the dynamic elemenst of the search state in general retured from ask */
        TParameters _parameters; /**< the optimizer's set of static parameters, from inputs or internal. */
        GradFunc _gfunc = nullptr; /**< gradient function, when available. */
        bool _initial_elitist = false; /**< restarts from and re-injects best seen solution if not the final one. */

    private:
        std::mt19937 _uhgen; /**< random device used for uncertainty handling operations. */
        std::uniform_real_distribution<> _uhunif;
        Eigen::EigenMultivariateNormal<double> _uhesolver;
    };


//template class ESOStrategy<CMAParameters<GenoPheno<NoBoundStrategy>>,CMASolutions >;
//template class ESOStrategy<CMAParameters<GenoPheno<pwqBoundStrategy>>,CMASolutions >;
//template class ESOStrategy<CMAParameters<GenoPheno<NoBoundStrategy,linScalingStrategy>>,CMASolutions >;
//template class ESOStrategy<CMAParameters<GenoPheno<pwqBoundStrategy,linScalingStrategy>>,CMASolutions >;



}
#endif
