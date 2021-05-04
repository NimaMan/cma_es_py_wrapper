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

#ifndef BIPOPCMASTRATEGY_H
#define BIPOPCMASTRATEGY_H

#include "ipopcmastrategy.h"
#include "llogging.h"
#include "opti_err.h"

#include <random>
#include <random>
#include <ctime>
#include <array>
namespace libcmaes
{
  /**
   * \brief Implementation of the BIPOP flavor of CMA-ES, with restarts that
   *        control the population of offsprings used in the update of the 
   *        distribution parameters in order to alternate between local and 
   *        global searches for the objective.
   */
  template <class TCovarianceUpdate, class TGenoPheno>
    class BIPOPCMAStrategy : public IPOPCMAStrategy<TCovarianceUpdate,TGenoPheno>
  {
  public:
    /**
     * \brief constructor.
     * @param parameters stochastic search parameters
     */
    BIPOPCMAStrategy( CMAParameters<TGenoPheno> &parameters)
    :IPOPCMAStrategy<TCovarianceUpdate,TGenoPheno>(parameters),_lambda_def(parameters._lambda),_lambda_l(parameters._lambda)
    {
        std::random_device rd;
        _gen = std::mt19937(rd());
        _gen.seed(static_cast<uint64_t>(time(nullptr)));
        _unif = std::uniform_real_distribution<>(0,1);
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._lambda = _lambda_def;
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._mu = floor(_lambda_def / 2.0);
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::_solutions = CMASolutions(CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters);
       _sigma_init = parameters._sigma_init;
        _max_fevals = parameters._max_fevals;
    };

    /**
     * \brief constructor.
     * @param parameters stochastic search parameters
     * @param solutions solution to start search from
     */
    BIPOPCMAStrategy(CMAParameters<TGenoPheno> &parameters,
		     const CMASolutions &solutions):IPOPCMAStrategy<TCovarianceUpdate,TGenoPheno>(parameters,solutions),_lambda_def(parameters._lambda),_lambda_l(parameters._lambda)
        {
          std::random_device rd;
          _gen = std::mt19937(rd());
          _gen.seed(static_cast<uint64_t>(time(nullptr)));
          _unif = std::uniform_real_distribution<>(0,1);
          CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._lambda = _lambda_def;
          CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._mu = floor(_lambda_def / 2.0);
          //CMAStrategy<TCovarianceUpdate,TGenoPheno>::_solutions = CMASolutions(CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters);
        };
    
    ~BIPOPCMAStrategy(){};

    /**
     * \brief Updates the covariance matrix and prepares for the next iteration.
     */
    
    void tell(const std::vector<double>& fvals){
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::tell(fvals);
    };


    

    
  protected:
    void r1(){
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._lambda = _lambda_l;
        IPOPCMAStrategy<TCovarianceUpdate,TGenoPheno>::lambda_inc();
        _lambda_l = CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._lambda;
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._sigma_init = _sigma_init;
    };
    void r2()  {
        double u = _unif(_gen);
        double us = _unif(_gen);
        double nsigma = 2.0*pow(10,-2.0*us);
        double ltmp = pow(0.5*(_lambda_l/_lambda_def),u);
        double nlambda = ceil(_lambda_def * ltmp);
        LOG_IF(INFO,!(CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._quiet)) << "Restart => lambda_s=" << nlambda << " / lambda_old=" << CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._lambda << " / lambda_l=" << _lambda_l << " / lambda_def=" << _lambda_def << " / nsigma=" << nsigma << std::endl;
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._lambda = nlambda;
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._sigma_init = nsigma;
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters.initialize_parameters();
    };

  private:
    std::mt19937 _gen;
    std::uniform_real_distribution<> _unif;
    double _lambda_def;
    double _lambda_l;
    double _sigma_init; // to save the original value
    double _max_fevals; // to save the original value
  };



  template class BIPOPCMAStrategy<CovarianceUpdate,GenoPheno<NoBoundStrategy>>;
  template class BIPOPCMAStrategy<ACovarianceUpdate,GenoPheno<NoBoundStrategy>>;
  template class BIPOPCMAStrategy<VDCMAUpdate,GenoPheno<NoBoundStrategy>>;
  template class BIPOPCMAStrategy<CovarianceUpdate,GenoPheno<pwqBoundStrategy>>;
  template class BIPOPCMAStrategy<ACovarianceUpdate,GenoPheno<pwqBoundStrategy>>;
  template class BIPOPCMAStrategy<VDCMAUpdate,GenoPheno<pwqBoundStrategy>>;
  template class BIPOPCMAStrategy<CovarianceUpdate,GenoPheno<NoBoundStrategy,linScalingStrategy>>;
  template class BIPOPCMAStrategy<ACovarianceUpdate,GenoPheno<NoBoundStrategy,linScalingStrategy>>;
  template class BIPOPCMAStrategy<VDCMAUpdate,GenoPheno<NoBoundStrategy,linScalingStrategy>>;
  template class BIPOPCMAStrategy<CovarianceUpdate,GenoPheno<pwqBoundStrategy,linScalingStrategy>>;
  template class BIPOPCMAStrategy<ACovarianceUpdate,GenoPheno<pwqBoundStrategy,linScalingStrategy>>;
  template class BIPOPCMAStrategy<VDCMAUpdate,GenoPheno<pwqBoundStrategy,linScalingStrategy>>;
}
#endif
