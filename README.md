# Project 5: GPU Computing with CUDA

## Warm-up

In this project,  you will write 2 CUDA kernels for doing heat diffusion in one
dimension. The first kernel will be a naive implementation, the second kernel
will leverage CUDA shared memory. Be sure to watch and read the material in the [associated pre-class assignments](../schedule.md)! Then, as you are developing and running your CUDA code, refer to the following ICER documentation pages for using GPUs on HPCC:

- [Compiling for GPUs](https://docs.icer.msu.edu/Compiling_for_GPUs/)
- [Requesting GPUs](https://docs.icer.msu.edu/Requesting_GPUs/)

I strongly recommend using HPCC for this project.

## Part 1

Write a naive implementation of the heat diffusion approximation in
CUDA, using the `host_diffusion` routine in the `diffusion.cu` starter code  as a guide.

All the lines that you will need to change have been denoted with a FIXME. In
addition to writing the kernel, you will also need to handle the allocation of
memory on the GPU, copying memory to and from the GPU, launching the CUDA
kernel, and freeing the GPU memory. Remember to consider how large your domain
is vs. how many points your kernel needs to run on. To make things easier, you
can assume that the domain size minus `2*NG` will be divisible by the block
size, which will be divisible by 32.

These kernels will be easiest to debug by running only for 10 time steps. I've
included a python file to plot the difference between the host kernel and CUDA
kernel for the first time step. Any differences between the two should look
randomized and small. When you think you have your kernel working, you can run
it for 1000000 steps (since this simple implementation of heat diffusion
converges slowly).

I've also included a debugging function in the C version that you can use to
wrap your calls to the CUDA library, such as `checkCuda(cudaMemcpy(...));`. You
can see were I got this and some examples of how this is used
[here](https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/finite-difference/finite-difference.cu).
You need to activate this debugging function at compile time by executing 
`nvcc diffusion.cu -DDEBUG -o diffusion`.

The CUDA blog posts on finite difference in
[C/C++](https://devblogs.nvidia.com/finite-difference-methods-cuda-cc-part-1/)
might also be useful.

**Done.**

## Part 2

Rewrite your naive implementation of the heat diffusion kernel to first load
from global memory into a buffer in shared memory.

It will probably be useful to have separate indices to keep track of a thread's
position with it's shared memory block and within the entire array. You will
need extra logic to load in ghost zones for each block, and some calls to
`__syncthreads()`. When you get to calculating the diffusion in the kernel, all
memory loads should be from shared memory.

This kernel should give identical results to the `cuda_diffusion` kernel.

**Done.**

## Part 3

Time your naive implementation, the shared memory implementation, and a case
where memory is copied on and off the GPU for every time step.

Uncomment and fill in the code to test `shared_diffusion` but with copying data
to and from the GPU between every time step. This is closer to how CUDA is
sometimes used in practice, when there is a heavily used portion of a large
legacy code that you wish to optimize.

Increase your grid size to `2^15+2*NG` and change the number of steps you take
to 100. Run the program and take note of the timings. 

**Done.**

## What to turn In

Your code, well commented, and answers to these questions:

1. Report your timings for the host, naive CUDA kernel, shared memory CUDA kernel,
and the excessive memory copying case, using block dimensions of 256, 512,
and 1024. Use a grid size of `2^15+2*NG` (or larger) and run for 100 steps (or
shorter, if it's taking too long). Remember to use `-O3`! 

    Below are the results using block dimensions of 256, 512 and 1024 with a grid size of 2^15+2*NG (32772 size) and 100 steps as requested:

    |Num Steps  | Grid Size  | Block Size | Host (ms) | Naive (ms) | Shared (ms) | Excessive Memcpy (ms) |
    |------------|------------|------------|----------|-----------|------------|----------------------|
    | 100 | 32772 | 256 | 0.154772 | 0.003210 | 0.003419 | 0.055540 |
    | 100 | 32772 | 512 | 0.157449 | 0.003290 | 0.003460 | 0.068581 |
    | 100 | 32772 | 1024 | 0.153420 | 0.003284 | 0.003376 | 0.053585 |

![part1](https://github.com/cmse822/project-5-gpus-with-cuda-team-6/assets/94200328/b9215917-aae1-4bda-ba3c-a2c2a87ed61e)

2. How do the GPU implementations compare to the single threaded host code. Is it
faster than the theoretical performance of the host if we used all the cores on
the CPU?

    As seen in the results table above, the GPU implementations are faster then the single threaded host. It is faster even when utilizing all the cores on the CPU. This probably is even more noticeable as the num steps increases.

3. For the naive kernel, the shared memory kernel, and the excessive `memcpy` case,
which is the slowest? Why? How might you design a larger code to avoid this slow down?

    The excessive memcpy case is the slowest. This is most likely due to the large number of data transfers from host to device. That is, the copying of data from host memory to device memory. This increasses overhead resulting in slower performance. For design purposes, I would try to minimize that data transfer between the host and device by creating variables on the device, aggregating the results, and transferring that data all back to the host at the end.

4. Do you see a slow down when you increase the block dimension? Why? Consider
that multiple blocks may run on a single multiprocessor simultaneously, sharing
the same shared memory.

    Not really, there's negligible slowdown and this is probably due to the I/O overhead masking most of paralellizable runtime performance differences between the block sizes. If the block sizes were so large that it fills up the the shared memory for the single multiprocessor, then there could be performance degradation that way, but I don't see that happening in this case.

## Plotting Results

Below are the plotting results we got:

**Fig 1**

![Figure 1](https://github.com/cmse822/project-5-gpus-with-cuda-team-6/blob/main/plots/fig1.png)

**Fig 2**

![Figure 2](https://github.com/cmse822/project-5-gpus-with-cuda-team-6/blob/main/plots/fig2.png)

**Fig 3**

![Figure 3](https://github.com/cmse822/project-5-gpus-with-cuda-team-6/blob/main/plots/fig3.png)


## How to run

Some notes on how to run `diffusion.cu`

Ran it on `dev-amd20-v100`:

```bash
    module purge
    module load NVHPC/21.9-GCCcore-10.3.0-CUDA-11.4
    nvcc diffusion.cu -DDEBUG -o diffusion -lstdc++fs -O3
    ./diffusion
```
