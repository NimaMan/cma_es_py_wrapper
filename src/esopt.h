

#include <vector>

#include "cmastrategy.h"

//#include "covarianceupdate.h"
//#include "acovarianceupdate.h"

//using namespace libcmaes;


class CMA{
    public: 
    CMA(std::vector<double> x0, double sigma):_x0(x0), _sigma(sigma){};
    ~CMA(){};

    libcmaes::CMAStrategy<libcmaes::CovarianceUpdate, libcmaes::GenoPheno<libcmaes::NoBoundStrategy>> create(){
          libcmaes::CMAParameters<> cmaparams(_x0, _sigma); 
          return libcmaes::CMAStrategy<libcmaes::CovarianceUpdate, libcmaes::GenoPheno<libcmaes::NoBoundStrategy>>(cmaparams); 
        };
    
    std::vector<double> _x0;
    double _sigma;
};