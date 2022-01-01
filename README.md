# CUDA Kernels
A collection of custom CUDA kernels.

--- 

## CorrelationMask

### Kernel Description

The primary inputs for this kernel are an `n` by `p` matrix and a *target* vector of `n` elements. The device kernel computes the Pearson correlation between the target vector and every column of the `n` by `p` input matrix. The kernel then checks if each Pearson correlation coefficient is within a user-defined range, returning a mask vector of `p` elements where each element is either a *0* (outside of range) or *1* (inside range).   

### Thread Layout

Because we expect `p >> n`, each thread is tasked with computing the Pearson correlation between *its column* of the input matrix and the *target* vector, and then determining whether *its rho* falls within the range specified. Hence, the kernel will invoke a `p` by `1` by `1` thread grid, where the number of 1-D thread blocks is `(p + 1023) // 1024`. 

---
