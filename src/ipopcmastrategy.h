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

#ifndef IPOPCMASTRATEGY_H
#define IPOPCMASTRATEGY_H

#include "cmastrategy.h"
#include "ipopcmastrategy.h"
#include "opti_err.h"
#include "llogging.h"

#include <iostream>

namespace libcmaes
{
  /**
   * \brief Implementation of the IPOP flavor of CMA-ES, with restarts
   *        that linearly increase the population of offsprings used in the 
   *        update of the distribution parameters.
   */
  template <class TCovarianceUpdate, class TGenoPheno>
    class IPOPCMAStrategy : public CMAStrategy<TCovarianceUpdate, TGenoPheno>
  {
  public:
    /**
     * \brief constructor.
     * @param parameters stochastic search parameters
     */
    IPOPCMAStrategy(CMAParameters<TGenoPheno> &parameters) 
    :CMAStrategy<TCovarianceUpdate,TGenoPheno>(parameters){};

    /**
     * \brief constructor.
     * @param parameters stochastic search parameters
     * @param solutions solution to start search from
     */
    IPOPCMAStrategy(CMAParameters<TGenoPheno> &parameters,
		    const CMASolutions &solutions):CMAStrategy<TCovarianceUpdate,TGenoPheno>(parameters,solutions){};
    
    ~IPOPCMAStrategy(){};

    /**
     * \brief Updates the covariance matrix and prepares for the next iteration.
     */
    void tell(const std::vector<double>& fvals){
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::tell(fvals);
    };
    

  protected:
    void lambda_inc(){
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._lambda *= 2.0;
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters.initialize_parameters();
        LOG_IF(INFO,!(CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._quiet)) << "Restart => lambda_l=" << CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._lambda << " / lambda_old=" << CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters._lambda / 2.0 << std::endl;
    };
    
    void reset_search_state(){
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::_solutions = CMASolutions(CMAStrategy<TCovarianceUpdate,TGenoPheno>::_parameters);
        CMAStrategy<TCovarianceUpdate,TGenoPheno>::_niter = 0;
    };
    
    void capture_best_solution(CMASolutions &best_run){
        if (best_run._candidates.empty() || CMAStrategy<TCovarianceUpdate,TGenoPheno>::_solutions.best_candidate().get_fvalue() < best_run.best_candidate().get_fvalue())
            best_run = CMAStrategy<TCovarianceUpdate,TGenoPheno>::_solutions;
    };
  
  };


  template class IPOPCMAStrategy<CovarianceUpdate,GenoPheno<NoBoundStrategy>>;
  template class IPOPCMAStrategy<ACovarianceUpdate,GenoPheno<NoBoundStrategy>>;
  template class IPOPCMAStrategy<VDCMAUpdate,GenoPheno<NoBoundStrategy>>;
  template class IPOPCMAStrategy<CovarianceUpdate,GenoPheno<pwqBoundStrategy>>;
  template class IPOPCMAStrategy<ACovarianceUpdate,GenoPheno<pwqBoundStrategy>>;
  template class IPOPCMAStrategy<VDCMAUpdate,GenoPheno<pwqBoundStrategy>>;
  template class IPOPCMAStrategy<CovarianceUpdate,GenoPheno<NoBoundStrategy,linScalingStrategy>>;
  template class IPOPCMAStrategy<ACovarianceUpdate,GenoPheno<NoBoundStrategy,linScalingStrategy>>;
  template class IPOPCMAStrategy<VDCMAUpdate,GenoPheno<NoBoundStrategy,linScalingStrategy>>;
  template class IPOPCMAStrategy<CovarianceUpdate,GenoPheno<pwqBoundStrategy,linScalingStrategy>>;
  template class IPOPCMAStrategy<ACovarianceUpdate,GenoPheno<pwqBoundStrategy,linScalingStrategy>>;
  template class IPOPCMAStrategy<VDCMAUpdate,GenoPheno<pwqBoundStrategy,linScalingStrategy>>;
}
#endif
