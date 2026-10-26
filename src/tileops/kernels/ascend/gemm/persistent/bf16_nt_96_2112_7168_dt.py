import tilelang
import tilelang.ascend.language as T


@T.macro
def swizzle(tile_idx):
    m_tile = T.alloc_var("int32")
    n_tile = T.alloc_var("int32")
    m_tile = tile_idx // 33
    n_tile = tile_idx % 33
    return (m_tile, n_tile)


@T.prim_func
def main(
    A: T.Tensor((96, 7168), "bfloat16"),
    B: T.Tensor((2112, 7168), "bfloat16"),
    D: T.Tensor((96, 2112), "bfloat16"),
):
    with T.Kernel(33) as bx:  # noqa: SIM117 - Preserve measured TileLang scopes.
        with T.Cube():
            a_l0 = T.alloc_l0a((2, 96, 128), "bfloat16")
            a_l1 = T.alloc_l1((2, 96, 256), "bfloat16")
            b_l0 = T.alloc_l0b((2, 64, 128), "bfloat16")
            b_l1 = T.alloc_l1((1, 64, 256), "bfloat16")
            c_l0 = T.alloc_l0c((96, 64), "float32")
            for tile_idx in T.Persistent([33], 33, bx):
                m_tile, n_tile = swizzle(tile_idx)
                T.assume(m_tile >= 0)
                T.assume(m_tile < 1)
                T.assume(n_tile >= 0)
                T.assume(n_tile < 33)
                m_base = m_tile * 96
                n_base = n_tile * 64
                T.copy(A[m_base : m_base + 96, 0:256], a_l1[0, :, :])
                T.ascend_set_flag("MTE2_MTE1", 0)
                T.copy(B[n_base : n_base + 64, 0:256], b_l1[0, :, :], l2_cache_ctrl="NOTALLOC_KEEP")
                T.ascend_set_flag("MTE2_MTE1", 2)
                T.ascend_wait_flag("MTE2_MTE1", 0)
                T.copy(a_l1[0, :, 128:256], a_l0[1, :, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.copy(a_l1[0, :, 0:128], a_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_set_flag("MTE1_MTE2", 0)
                T.copy(A[m_base : m_base + 96, 256:512], a_l1[1, :, :])
                T.ascend_set_flag("MTE2_MTE1", 1)
                T.ascend_wait_flag("MTE2_MTE1", 2)
                T.copy(b_l1[0, :, 0:128], b_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.copy(b_l1[0, :, 128:256], b_l0[1, :, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_set_flag("MTE1_MTE2", 2)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(a_l0[0, :, :], b_l0[0, :, :], c_l0, transpose_B=True, clear_accum=True)
                for cycle_0 in range(12):
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(a_l0[1, :, :], b_l0[1, :, :], c_l0, transpose_B=True, clear_accum=False)
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(
                        B[n_base : n_base + 64, 256 + 512 * cycle_0 : 512 + 512 * cycle_0],
                        b_l1[0, :, :],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, :, 128:256], a_l0[1, :, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, :, 0:128], a_l0[0, :, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(
                        A[m_base : m_base + 96, 512 + 512 * cycle_0 : 768 + 512 * cycle_0],
                        a_l1[0, :, :],
                    )
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(b_l1[0, :, 0:128], b_l0[0, :, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.copy(b_l1[0, :, 128:256], b_l0[1, :, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(a_l0[0, :, :], b_l0[0, :, :], c_l0, transpose_B=True, clear_accum=False)
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(a_l0[1, :, :], b_l0[1, :, :], c_l0, transpose_B=True, clear_accum=False)
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(
                        B[n_base : n_base + 64, 512 + 512 * cycle_0 : 768 + 512 * cycle_0],
                        b_l1[0, :, :],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, :, 128:256], a_l0[1, :, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, :, 0:128], a_l0[0, :, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(
                        A[m_base : m_base + 96, 768 + 512 * cycle_0 : 1024 + 512 * cycle_0],
                        a_l1[1, :, :],
                    )
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(b_l1[0, :, 0:128], b_l0[0, :, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.copy(b_l1[0, :, 128:256], b_l0[1, :, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(a_l0[0, :, :], b_l0[0, :, :], c_l0, transpose_B=True, clear_accum=False)
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(a_l0[1, :, :], b_l0[1, :, :], c_l0, transpose_B=True, clear_accum=False)
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("MTE1_MTE2", 2)
                T.copy(
                    B[n_base : n_base + 64, 6400:6656], b_l1[0, :, :], l2_cache_ctrl="NOTALLOC_KEEP"
                )
                T.ascend_set_flag("MTE2_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 1)
                T.ascend_wait_flag("MTE2_MTE1", 1)
                T.copy(a_l1[1, :, 128:256], a_l0[1, :, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_wait_flag("M_MTE1", 0)
                T.copy(a_l1[1, :, 0:128], a_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_set_flag("MTE1_MTE2", 1)
                T.ascend_wait_flag("MTE1_MTE2", 0)
                T.copy(A[m_base : m_base + 96, 6656:6912], a_l1[0, :, :])
                T.ascend_set_flag("MTE2_MTE1", 0)
                T.ascend_wait_flag("M_MTE1", 2)
                T.ascend_wait_flag("MTE2_MTE1", 2)
                T.copy(b_l1[0, :, 0:128], b_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[0, :, 128:256], b_l0[1, :, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_set_flag("MTE1_MTE2", 2)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(a_l0[0, :, :], b_l0[0, :, :], c_l0, transpose_B=True, clear_accum=False)
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(a_l0[1, :, :], b_l0[1, :, :], c_l0, transpose_B=True, clear_accum=False)
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("MTE1_MTE2", 2)
                T.copy(
                    B[n_base : n_base + 64, 6656:6912], b_l1[0, :, :], l2_cache_ctrl="NOTALLOC_KEEP"
                )
                T.ascend_set_flag("MTE2_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 1)
                T.ascend_wait_flag("MTE2_MTE1", 0)
                T.copy(a_l1[0, :, 128:256], a_l0[1, :, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_wait_flag("M_MTE1", 0)
                T.copy(a_l1[0, :, 0:128], a_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_MTE2", 1)
                T.copy(A[m_base : m_base + 96, 6912:7168], a_l1[1, :, :])
                T.ascend_set_flag("MTE2_MTE1", 1)
                T.ascend_wait_flag("M_MTE1", 2)
                T.ascend_wait_flag("MTE2_MTE1", 2)
                T.copy(b_l1[0, :, 0:128], b_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[0, :, 128:256], b_l0[1, :, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_set_flag("MTE1_MTE2", 2)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(a_l0[0, :, :], b_l0[0, :, :], c_l0, transpose_B=True, clear_accum=False)
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(a_l0[1, :, :], b_l0[1, :, :], c_l0, transpose_B=True, clear_accum=False)
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("MTE1_MTE2", 2)
                T.copy(
                    B[n_base : n_base + 64, 6912:7168], b_l1[0, :, :], l2_cache_ctrl="NOTALLOC_KEEP"
                )
                T.ascend_set_flag("MTE2_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 1)
                T.ascend_wait_flag("MTE2_MTE1", 1)
                T.copy(a_l1[1, :, 128:256], a_l0[1, :, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_wait_flag("M_MTE1", 0)
                T.copy(a_l1[1, :, 0:128], a_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("M_MTE1", 2)
                T.ascend_wait_flag("MTE2_MTE1", 2)
                T.copy(b_l1[0, :, 0:128], b_l0[0, :, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[0, :, 128:256], b_l0[1, :, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(a_l0[0, :, :], b_l0[0, :, :], c_l0, transpose_B=True, clear_accum=False)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(a_l0[1, :, :], b_l0[1, :, :], c_l0, transpose_B=True, clear_accum=False)
                T.ascend_set_flag("M_FIX", 0)
                T.ascend_wait_flag("M_FIX", 0)
                T.copy(c_l0, D[m_base : m_base + 96, n_base : n_base + 64])
                T.ascend_pipe_barrier("PIPE_ALL")


def compile_kernel():
    return tilelang.compile(
        main,
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_auto_schedule": False, "tl.disable_shared_memory_reuse": True},
    )
