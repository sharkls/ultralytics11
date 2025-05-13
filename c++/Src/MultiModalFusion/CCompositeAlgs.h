#ifndef CCOMPOSITEALGS_MULTIMODALFUSION_H
#define CCOMPOSITEALGS_MULTIMODALFUSION_H

#include <iostream>
#include <memory>
#include <vector>
// #include <xtensor/xview.hpp>
// #include <xtensor/xnpy.hpp>
// #include <xtensor/xsort.hpp>
// #include <xtensor/xarray.hpp>
#include "ICommonAlg.h"
#include "IMakeAlg.h"
#include "CMakeAlgs.h"

class CCompositeAlgs final : public ICommonAlg {
public:
    CCompositeAlgs() = delete;
    explicit CCompositeAlgs(IMakeAlgPtr& p_makeAlgs) {
        m_vecAlgorithms = p_makeAlgs->getAlgorithms(); // 将4个模块待执行的部分放入m_vecAlgorithms中
    }

    void init(CSelfAlgParam* p_pAlgParam) override {
        for (const auto& p_algorithm : m_vecAlgorithms) {
            p_algorithm->setCommonData(m_pCommonData);
            p_algorithm->init(p_pAlgParam);
            m_pCommonData = p_algorithm->getCommonData();
        }
    }

    void execute() override {
        for (const auto& p_algorithm : m_vecAlgorithms) {
            p_algorithm->execute();
        }
    }
    
    void setCommonData(CCommonDataPtr p_commonData) override {
        m_pCommonData = p_commonData;
    }

    CCommonDataPtr getCommonData() override {
        return m_pCommonData;
    }
    
    void setCommonAllData(CCommonDataPtr p_commonData) {
        setCommonData(p_commonData);
        for (const auto& p_algorithm : m_vecAlgorithms) {
            p_algorithm->setCommonData(m_pCommonData);
        }
    }

private:
    std::vector<ICommonAlgPtr> m_vecAlgorithms;
};

using CCompositeAlgsPtr = std::shared_ptr<CCompositeAlgs>;
#endif //CCOMPOSITEALGS_MULTIMODALFUSION_H
