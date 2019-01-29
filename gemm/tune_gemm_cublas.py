#!/usr/bin/env python
import logging
import numpy as np

from kernel_tuner import run_kernel


def test():

    cp = ["-I/home/bwn200/eigen-git-mirror/", "-I/home/bwn200/cxxopts/include/", "-lcublas", "-lcurand"]

    size = np.int32(1024)
    problem_size = (size, size)

    #C program assumes data is stored column-major
    A = np.random.randn(*problem_size).astype(np.float32, order='F')
    B = np.random.randn(*problem_size).astype(np.float32, order='F')
    C = np.zeros_like(A)

    args = [C, A, B, size]

    answer = run_kernel("call_cublas_gemm_basic_version", "gemm_cublas.cpp",
                        1, args, params={},
                        compiler_options=cp, compiler="nvcc", lang="C", log=logging.DEBUG)

    #numpy insists on returning the result in row-major, regardless of input
    #using a transpose as a quick fix, there should be a better solution
    expected = np.dot(A,B).T

    assert np.allclose(expected, answer[0], atol=1e-3)





if __name__ == "__main__":
    test()


