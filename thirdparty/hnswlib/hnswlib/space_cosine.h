#pragma once

#include "hnswlib.h"
#include "simd/hook.h"

namespace hnswlib {

static float
Cosine(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return faiss::fvec_inner_product((const float*)pVect1, (const float*)pVect2, *((size_t*)qty_ptr));
}

static float
CosineDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -1.0f * Cosine(pVect1, pVect2, qty_ptr);
}

class CosineSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    CosineSpace(size_t dim) {
        fstdistfunc_ = CosineDistance;
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t
    get_data_size() {
        return data_size_;
    }

    DISTFUNC<float>
    get_dist_func() {
        return fstdistfunc_;
    }

    void*
    get_dist_func_param() {
        return &dim_;
    }

    ~CosineSpace() {
    }
};

static knowhere::fp16
CosineFP16(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return faiss::fvec_inner_product((const float*)pVect1, (const float*)pVect2, *((size_t*)qty_ptr));
}

static knowhere::fp16
CosineDistanceFP16(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -1.0f * CosineFP16(pVect1, pVect2, qty_ptr);
}

class CosineSpaceFP16 : public SpaceInterface<knowhere::fp16> {
    DISTFUNC<knowhere::fp16> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    CosineSpaceFP16(size_t dim) {
        fstdistfunc_ = CosineDistanceFP16;
        dim_ = dim;
        data_size_ = dim * sizeof(knowhere::fp16);
    }

    size_t
    get_data_size() {
        return data_size_;
    }

    DISTFUNC<knowhere::fp16>
    get_dist_func() {
        return fstdistfunc_;
    }

    void*
    get_dist_func_param() {
        return &dim_;
    }

    ~CosineSpaceFP16() {
    }
};

static knowhere::bf16
CosineBF16(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    // return faiss::fvec_inner_product((const float*)pVect1, (const float*)pVect2, *((size_t*)qty_ptr));
    knowhere::bf16* pVect1v = (knowhere::bf16*)pVect1;
    knowhere::bf16* pVect2v = (knowhere::bf16*)pVect2;
    // 将bf16转换为float
    std::vector<float> pVect1_float;
    std::vector<float> pVect2_float;

    for (size_t i = 0; i < *((size_t*)qty_ptr); i++) {
        pVect1_float.push_back((float)pVect1v[i]);
        pVect2_float.push_back((float)pVect2v[i]);
        pVect1v++;
        pVect2v++;
    }

    return (knowhere::bf16)faiss::fvec_inner_product(pVect1_float.data(), pVect2_float.data(), *((size_t*)qty_ptr));
}

static knowhere::bf16
CosineDistanceBF16(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return -1.0f * CosineBF16(pVect1, pVect2, qty_ptr);
}

class CosineSpaceBF16 : public SpaceInterface<knowhere::bf16> {
    DISTFUNC<knowhere::bf16> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    CosineSpaceBF16(size_t dim) {
        fstdistfunc_ = CosineDistanceBF16;
        dim_ = dim;
        data_size_ = dim * sizeof(knowhere::bf16);
    }

    size_t
    get_data_size() {
        return data_size_;
    }

    DISTFUNC<knowhere::bf16>
    get_dist_func() {
        return fstdistfunc_;
    }

    void*
    get_dist_func_param() {
        return &dim_;
    }

    ~CosineSpaceBF16() {
    }
};

}  // namespace hnswlib
