# CUDA Kernels

## async_image_normalize

### Kernel Description

Normalize images (in this simplified case a 1-D representation of a 2-D image) to values between 0 and 1.

### Program Workflow

The image is first written into a host staging area that consists of WriteCombined memory. From there, it is asynchronously copied to the device staging area, normalized, and then asynchronously sent back to the host. The particular event-based, across-stream synchronization scheme that I have employed is very heavy, so the images need to be relatively large for it to perform well. As such, this program reinforces the notion that memory usage and program workflow often should be organized so as to minimize necessary host-device synchronization.

---

## curve_length

### Kernel Description

A definition of the *curve length* algorithm can be found [here](https://lcp.mit.edu/pdf/Zong06.pdf).

### Thread Layout

The primary input for this kernel is an `n` by `p` matrix where `n` represents the number of signals and `p` represents the length of each signal. Each thread block is responsible for computing the curve length of at least 1 signal (> 1 if the number of signals exceeds 65 000). As such, the kernel returns a vector of size `n` of curve lengths.

---

## f32_reduce

### Kernel Description

Performs a summation reduction on a vector of 32-bit floats. 

### Thread Layout

Each thread block is responsible for computing a partial sum of the input vector. All of the benefits of GPU parallelization are hidden in this case due to cumbersome memory transfers. As such, the summation should be used primarily as a device function.

---

## correlation_mask

### Kernel Description

The primary inputs for this kernel are an `n` by `p` matrix and a *target* vector of `n` elements. The device kernel computes the Pearson correlation between the target vector and every column of the `n` by `p` input matrix. The kernel then checks if each Pearson correlation coefficient is within a user-defined range, returning a mask vector of `p` elements where each element is either a *0* (outside of range) or *1* (inside range).   

### Thread Layout

Because we expect `p >> n`, each thread is tasked with computing the Pearson correlation between *its column* of the input matrix and the *target* vector, and then determining whether *its rho* falls within the range specified. Hence, the kernel will invoke a `p` by `1` by `1` thread grid, where the number of 1-D thread blocks is `(p + 1023) // 1024`. 

---
