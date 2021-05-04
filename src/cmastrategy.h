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

#ifndef CMASTRATEGY_H
#define CMASTRATEGY_H

#include "esostrategy.h"
#include "cmaparameters.h"
#include "cmasolutions.h"
#include "covarianceupdate.h"
#include "acovarianceupdate.h"
#include "vdcmaupdate.h"
#include "eigenmvn.h"
#include <fstream>


#include "libcmaes_config.h"
#include "opti_err.h"
#include "llogging.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>

namespace libcmaes
{

  /**
   * \brief This is an implementation of CMA-ES. 
   */
  template <class TCovarianceUpdate,class TGenoPheno=GenoPheno<NoBoundStrategy> >
    class CMAStrategy : public ESOStrategy<CMAParameters<TGenoPheno>,CMASolutions >
    {
    public:
        /**
        * \brief dummy constructor
        */
        CMAStrategy():ESOStrategy<CMAParameters<TGenoPheno>,CMASolutions >(){};
        /**
         * \brief constructor.
         * @param parameters stochastic search parameters
         */
        CMAStrategy(CMAParameters<TGenoPheno> &parameters) :ESOStrategy<CMAParameters<TGenoPheno>,CMASolutions >(parameters){
            _esolver = Eigen::EigenMultivariateNormal<double>(false, this->_parameters._seed); // seeding the multivariate normal generator.
            LOG_IF(INFO,!this->_parameters._quiet) << "CMA-ES / dim=" << this->_parameters._dim << " / lambda=" << this->_parameters._lambda << " / sigma0=" << this->_solutions._sigma << " / mu=" << this->_parameters._mu << " / mueff=" << this->_parameters._muw << " / c1=" << this->_parameters._c1 << " / cmu=" <<this->_parameters._cmu << " / tpa=" << (this->_parameters._tpa==2) << " / threads=" << Eigen::nbThreads() << std::endl;
          
        };

        /**
        * \brief constructor for starting from an existing solution.
        * @param parameters stochastic search parameters
        * @param cmasols solution object to start from
        */
        
        CMAStrategy(CMAParameters<TGenoPheno> &parameters, const CMASolutions &cmasols)
        :ESOStrategy<CMAParameters<TGenoPheno>,CMASolutions >(parameters,cmasols){
        
            _esolver = Eigen::EigenMultivariateNormal<double>(false,this->_parameters._seed); // seeding the multivariate normal generator.
            LOG_IF(INFO,!this->_parameters._quiet) << "CMA-ES / dim=" << this->_parameters._dim << " / lambda=" << this->_parameters._lambda << " / sigma0=" << this->_solutions._sigma << " / mu=" << this->_parameters._mu << " / mueff=" << this->_parameters._muw << " / c1=" << this->_parameters._c1 << " / cmu=" << this->_parameters._cmu << " / lazy_update=" << this->_parameters._lazy_update << std::endl;

        };

        ~CMAStrategy(){ };

        /**
         * \brief generates nsols new candidate solutions, sampled from a 
         *        multivariate normal distribution.
         * return A matrix whose rows contain the candidate points.
         */
        dMat ask(){
            #ifdef HAVE_DEBUG
                std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
            #endif
            
            // compute eigenvalues and eigenvectors.
            if (!this->_parameters._sep && !this->_parameters._vd){
                this->_solutions._updated_eigen = false;
                if (this->_niter == 0 || !this->_parameters._lazy_update 
                    || this->_niter - this->_solutions._eigeniter > this->_parameters._lazy_value){
                    this->_solutions._eigeniter = this->_niter;
                    _esolver.setMean(this->_solutions._xmean);
                    _esolver.setCovar(this->_solutions._cov);
                    this->_solutions._updated_eigen = true;
                }
            }
            else if (this->_parameters._sep){
              _esolver.setMean(this->_solutions._xmean);
              _esolver.set_covar(this->_solutions._sepcov);
              _esolver.set_transform(this->_solutions._sepcov.cwiseSqrt());
            }
            else if (this->_parameters._vd){
              _esolver.setMean(this->_solutions._xmean);
              _esolver.set_covar(this->_solutions._sepcov);
            }

            //debug
            //std::cout << "transform: " << _esolver._transform << std::endl;
            //debug
            
            // sample for multivariate normal distribution, produces one candidate per column.
            dMat pop;
            if (!this->_parameters._sep && !this->_parameters._vd)
                pop = _esolver.samples(this->_parameters._lambda,this->_solutions._sigma); // Eq (1).
            else if (this->_parameters._sep)
                pop = _esolver.samples_ind(this->_parameters._lambda,this->_solutions._sigma);
            else if (this->_parameters._vd){
                pop = _esolver.samples_ind(this->_parameters._lambda);
                double normv = this->_solutions._v.squaredNorm();
                double fact = std::sqrt(1+normv)-1;
                dVec vbar = this->_solutions._v / std::sqrt(normv);

                pop += fact * vbar * (vbar.transpose() * pop);
                for (int i=0;i<pop.cols();i++){
                    pop.col(i) = this->_solutions._xmean + this->_solutions._sigma * this->_solutions._sepcov.cwiseProduct(pop.col(i));
                }
            }
          
            // gradient if available.
            /*
            if (this->_parameters._with_gradient) {
                dVec grad_at_mean = this->gradf(this->_parameters._gp.pheno(this->_solutions._xmean));
                dVec gradgp_at_mean = this->gradgp(this->_solutions._xmean); // for geno / pheno transform.
                grad_at_mean = grad_at_mean.cwiseProduct(gradgp_at_mean);
                if (grad_at_mean != dVec::Zero(this->_parameters._dim)){
                    dVec nx;
                    if (!this->_parameters._sep && !this->_parameters._vd){
                        dMat sqrtcov = _esolver._eigenSolver.operatorSqrt();
                        dVec q = sqrtcov * grad_at_mean;
                        double normq = q.squaredNorm();
                        nx = this->_solutions._xmean - this->_solutions._sigma * (sqrt(this->_parameters._dim / normq)) * this->_solutions._cov * grad_at_mean;
                    }
                    else nx = this->_solutions._xmean - this->_solutions._sigma * (sqrt(this->_parameters._dim) / ((this->_solutions._sepcov.cwiseSqrt().cwiseProduct(grad_at_mean)).norm())) * this->_solutions._sepcov.cwiseProduct(grad_at_mean);
                    pop.col(0) = nx;
                }
            }
            */

          // tpa: fill up two first (or second in case of gradient) points with candidates usable for tpa computation
          if (this->_parameters._tpa == 2  && this->_niter > 0)
            {
        dVec mean_shift = this->_solutions._xmean - this->_solutions._xmean_prev;
        double mean_shift_norm = 1.0;
        if (!this->_parameters._sep && !this->_parameters._vd)
          mean_shift_norm = (_esolver._eigenSolver.eigenvalues().cwiseSqrt().cwiseInverse().cwiseProduct(_esolver._eigenSolver.eigenvectors().transpose()*mean_shift)).norm() / this->_solutions._sigma;
        else mean_shift_norm = this->_solutions._sepcov.cwiseSqrt().cwiseInverse().cwiseProduct(mean_shift).norm() / this->_solutions._sigma;
        //std::cout << "mean_shift_norm=" << mean_shift_norm << " / sqrt(N)=" << std::sqrt(std::sqrt(this->_parameters._dim)) << std::endl;

        dMat rz = _esolver.samples_ind(1);
        double mfactor = rz.norm();
        dVec z = mfactor * (mean_shift / mean_shift_norm);
        this->_solutions._tpa_x1 = this->_solutions._xmean + z;
        this->_solutions._tpa_x2 = this->_solutions._xmean - z;
        
        // if gradient is in col 0, move tpa vectors to pos 1 & 2
        if (this->_parameters._with_gradient)
          {
            this->_solutions._tpa_p1 = 1;
            this->_solutions._tpa_p2 = 2;
          }
        pop.col(this->_solutions._tpa_p1) = this->_solutions._tpa_x1;
        pop.col(this->_solutions._tpa_p2) = this->_solutions._tpa_x2;
            }
          
          // if some parameters are fixed, reset them.
          if (!this->_parameters._fixed_p.empty()){
              for (auto it=this->_parameters._fixed_p.begin();
                it!=this->_parameters._fixed_p.end();++it){
                pop.block((*it).first,0,1,pop.cols()) = dVec::Constant(pop.cols(),(*it).second).transpose();
              }
          }
          
          //debug
          /*DLOG(INFO) << "ask: produced " << pop.cols() << " candidates\n";
            std::cerr << pop << std::endl;*/
          //debug

        #ifdef HAVE_DEBUG
            std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
            this->_solutions._elapsed_ask = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
        #endif
          this->_currentCandidates = pop;
          return pop;
      };

        /**
         * \brief Updates the covariance matrix and prepares for the next iteration.
         */
        void tell(const std::vector<double>& fvals){
      //debug
      //DLOG(INFO) << "tell()\n";
      //debug

      /* 
      Set the fvalues of each candidate back to the solution candidates
      */

      
      // custom eval.
      for (int r=0;r<this->_currentCandidates.cols();r++){
          this->_solutions.get_candidate(r).set_x(this->_currentCandidates.col(r));
          // if (phenocandidates.size()) // if candidates in phenotype space are given
          //  this->_solutions.get_candidate(r).set_fvalue(_func(phenocandidates.col(r).data(), candidates.rows()));
          //else
          double f = fvals[r];
          this->_solutions.get_candidate(r).set_fvalue(f);
          
          //std::cerr << "candidate x: " << this->_solutions.get_candidate(r).get_x_dvec().transpose() << std::endl;
      }
      //update_fevals(candidates.cols());

  
      #ifdef HAVE_DEBUG
          std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
      #endif
    
      // sort candidates.
      if (!this->_parameters._uh)
        this->_solutions.sort_candidates();
      else this->uncertainty_handling();
      
      // call on tpa computation of s(t)
      if (this->_parameters._tpa == 2 && this->_niter > 0)
        this->tpa_update();

      // update function value history, as needed.
      this->_solutions.update_best_candidates();
    
      // CMA-ES update, depends on the selected 'flavor'.
      TCovarianceUpdate::update(this->_parameters,_esolver,this->_solutions);
    
      if (this->_parameters._uh)
        if (this->_solutions._suh > 0.0)
    this->_solutions._sigma *= this->_parameters._alphathuh;

    // other stuff.
    if (!this->_parameters._sep && !this->_parameters._vd)
      this->_solutions.update_eigenv(_esolver._eigenSolver.eigenvalues(),
						    _esolver._eigenSolver.eigenvectors());
    else this->_solutions.update_eigenv(this->_solutions._sepcov,
						       dMat::Constant(this->_parameters._dim,1,1.0));

    #ifdef HAVE_DEBUG
        std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
        this->_solutions._elapsed_tell = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
    #endif
    this->inc_iter(); // important step: signals next iteration.
  };
      
    protected:
      Eigen::EigenMultivariateNormal<double> _esolver;  /**< multivariate normal distribution sampler, and eigendecomposition solver. */
      std::ofstream *_fplotstream = nullptr; /**< plotting file stream, not in parameters because of copy-constructor hell. */
    
    };

  
}
#endif
