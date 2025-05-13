#ifndef IMAKEALG_MULTIMODALFUSION_H
#define IMAKEALG_MULTIMODALFUSION_H

#include "ICommonAlg.h"

class IMakeAlg {
public:
    IMakeAlg() = default;
    ~IMakeAlg() = default;

    virtual void makePre() = 0;
    virtual void makeInf() = 0;
    virtual void makePost() = 0;
    virtual void makeAlgs() = 0;
    
    std::vector<ICommonAlgPtr> getAlgorithms()
    {
        return m_vecAlgorithms;
    }

public:
    ICommonAlgPtr m_pPreAlgorithm;
    ICommonAlgPtr m_pInferenceAlgorithm;
    ICommonAlgPtr m_pPostAlgorithm;
    std::vector<ICommonAlgPtr> m_vecAlgorithms;
};

using IMakeAlgPtr = std::shared_ptr<IMakeAlg>;

#endif //IMAKEALG_MULTIMODALFUSION_H