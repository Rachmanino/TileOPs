import tilelang
import tilelang.ascend.language as T


@T.macro
def swizzle(tile_idx):
    m_tile = T.alloc_var("int32")
    n_tile = T.alloc_var("int32")
    m_tile = tile_idx // 35
    n_tile = tile_idx % 35
    return (m_tile, n_tile)


@T.prim_func
def main(
    A: T.Tensor((64, 2048), "bfloat16"),
    B: T.Tensor((7168, 2048), "bfloat16"),
    D: T.Tensor((64, 7168), "bfloat16"),
):
    with T.Kernel(35) as bx:  # noqa: SIM117 - Preserve measured TileLang scopes.
        with T.Cube():
            a_l0 = T.alloc_l0a((2, 64, 64), "bfloat16")
            a_l1 = T.alloc_l1((2, 64, 128), "bfloat16")
            b_l0 = T.alloc_l0b((2, 208, 64), "bfloat16")
            b_l1 = T.alloc_l1((2, 208, 128), "bfloat16")
            c_l0 = T.alloc_l0c((64, 208), "float32")
            for tile_idx in T.Persistent([35], 35, bx):
                m_tile, n_tile = swizzle(tile_idx)
                T.assume(m_tile >= 0)
                T.assume(m_tile < 1)
                T.assume(n_tile >= 0)
                T.assume(n_tile < 35)
                m_base = m_tile * 64
                n_base = n_tile * 208
                if n_tile < 34:
                    T.copy(
                        B[n_base : n_base + 208, 0:128],
                        b_l1[0, :208, :],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.copy(A[m_base : m_base + 64, 0:128], a_l1[0, :64, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.copy(
                        B[n_base : n_base + 208, 128:256],
                        b_l1[1, :208, :],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(b_l1[0, 0:208, 0:64], b_l0[0, :208, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:64, 0:64], a_l0[0, :64, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.copy(b_l1[0, 0:208, 64:128], b_l0[1, :208, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[0, :64, :],
                        b_l0[0, :208, :],
                        c_l0[:64, :208],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 2)
                    for cycle_0 in range(6):
                        T.copy(a_l1[0, 0:64, 64:128], a_l0[1, :64, :])
                        T.ascend_set_flag("MTE1_M", 1)
                        T.ascend_set_flag("MTE1_MTE2", 0)
                        T.ascend_wait_flag("MTE1_M", 1)
                        T.ascend_wait_flag("MTE1_M", 3)
                        T.gemm(
                            a_l0[1, :64, :],
                            b_l0[1, :208, :],
                            c_l0[:64, :208],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 1)
                        T.ascend_set_flag("M_MTE1", 3)
                        T.ascend_wait_flag("MTE1_MTE2", 0)
                        T.copy(
                            A[m_base : m_base + 64, 128 + 256 * cycle_0 : 256 + 256 * cycle_0],
                            a_l1[0, :64, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 0)
                        T.ascend_wait_flag("MTE1_MTE2", 1)
                        T.copy(
                            B[n_base : n_base + 208, 256 + 256 * cycle_0 : 384 + 256 * cycle_0],
                            b_l1[0, :208, :],
                            l2_cache_ctrl="NOTALLOC_KEEP",
                        )
                        T.ascend_set_flag("MTE2_MTE1", 1)
                        T.ascend_wait_flag("M_MTE1", 2)
                        T.ascend_wait_flag("MTE2_MTE1", 2)
                        T.copy(b_l1[1, 0:208, 0:64], b_l0[0, :208, :])
                        T.ascend_set_flag("MTE1_M", 2)
                        T.ascend_wait_flag("M_MTE1", 0)
                        T.ascend_wait_flag("MTE2_MTE1", 0)
                        T.copy(a_l1[0, 0:64, 0:64], a_l0[0, :64, :])
                        T.ascend_set_flag("MTE1_M", 0)
                        T.ascend_wait_flag("M_MTE1", 3)
                        T.copy(b_l1[1, 0:208, 64:128], b_l0[1, :208, :])
                        T.ascend_set_flag("MTE1_M", 3)
                        T.ascend_set_flag("MTE1_MTE2", 2)
                        T.ascend_wait_flag("MTE1_M", 0)
                        T.ascend_wait_flag("MTE1_M", 2)
                        T.gemm(
                            a_l0[0, :64, :],
                            b_l0[0, :208, :],
                            c_l0[:64, :208],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 0)
                        T.ascend_set_flag("M_MTE1", 2)
                        T.ascend_wait_flag("M_MTE1", 1)
                        T.copy(a_l1[0, 0:64, 64:128], a_l0[1, :64, :])
                        T.ascend_set_flag("MTE1_M", 1)
                        T.ascend_set_flag("MTE1_MTE2", 0)
                        T.ascend_wait_flag("MTE1_M", 1)
                        T.ascend_wait_flag("MTE1_M", 3)
                        T.gemm(
                            a_l0[1, :64, :],
                            b_l0[1, :208, :],
                            c_l0[:64, :208],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 1)
                        T.ascend_set_flag("M_MTE1", 3)
                        T.ascend_wait_flag("MTE1_MTE2", 0)
                        T.copy(
                            A[m_base : m_base + 64, 256 + 256 * cycle_0 : 384 + 256 * cycle_0],
                            a_l1[0, :64, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 0)
                        T.ascend_wait_flag("MTE1_MTE2", 2)
                        T.copy(
                            B[n_base : n_base + 208, 384 + 256 * cycle_0 : 512 + 256 * cycle_0],
                            b_l1[1, :208, :],
                            l2_cache_ctrl="NOTALLOC_KEEP",
                        )
                        T.ascend_set_flag("MTE2_MTE1", 2)
                        T.ascend_wait_flag("M_MTE1", 2)
                        T.ascend_wait_flag("MTE2_MTE1", 1)
                        T.copy(b_l1[0, 0:208, 0:64], b_l0[0, :208, :])
                        T.ascend_set_flag("MTE1_M", 2)
                        T.ascend_wait_flag("M_MTE1", 0)
                        T.ascend_wait_flag("MTE2_MTE1", 0)
                        T.copy(a_l1[0, 0:64, 0:64], a_l0[0, :64, :])
                        T.ascend_set_flag("MTE1_M", 0)
                        T.ascend_wait_flag("M_MTE1", 3)
                        T.copy(b_l1[0, 0:208, 64:128], b_l0[1, :208, :])
                        T.ascend_set_flag("MTE1_M", 3)
                        T.ascend_set_flag("MTE1_MTE2", 1)
                        T.ascend_wait_flag("MTE1_M", 0)
                        T.ascend_wait_flag("MTE1_M", 2)
                        T.gemm(
                            a_l0[0, :64, :],
                            b_l0[0, :208, :],
                            c_l0[:64, :208],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 0)
                        T.ascend_set_flag("M_MTE1", 2)
                        T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 0:64, 64:128], a_l0[1, :64, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :64, :],
                        b_l0[1, :208, :],
                        c_l0[:64, :208],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 64, 1664:1792], a_l1[0, :64, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(
                        B[n_base : n_base + 208, 1792:1920],
                        b_l1[0, :208, :],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(b_l1[1, 0:208, 0:64], b_l0[0, :208, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:64, 0:64], a_l0[0, :64, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.copy(b_l1[1, 0:208, 64:128], b_l0[1, :208, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[0, :64, :],
                        b_l0[0, :208, :],
                        c_l0[:64, :208],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 0:64, 64:128], a_l0[1, :64, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :64, :],
                        b_l0[1, :208, :],
                        c_l0[:64, :208],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 64, 1792:1920], a_l1[0, :64, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(
                        B[n_base : n_base + 208, 1920:2048],
                        b_l1[1, :208, :],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(b_l1[0, 0:208, 0:64], b_l0[0, :208, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:64, 0:64], a_l0[0, :64, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.copy(b_l1[0, 0:208, 64:128], b_l0[1, :208, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[0, :64, :],
                        b_l0[0, :208, :],
                        c_l0[:64, :208],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 0:64, 64:128], a_l0[1, :64, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :64, :],
                        b_l0[1, :208, :],
                        c_l0[:64, :208],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 64, 1920:2048], a_l1[0, :64, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(b_l1[1, 0:208, 0:64], b_l0[0, :208, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:64, 0:64], a_l0[0, :64, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.copy(b_l1[1, 0:208, 64:128], b_l0[1, :208, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[0, :64, :],
                        b_l0[0, :208, :],
                        c_l0[:64, :208],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 0:64, 64:128], a_l0[1, :64, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :64, :],
                        b_l0[1, :208, :],
                        c_l0[:64, :208],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_FIX", 0)
                    T.ascend_wait_flag("M_FIX", 0)
                    T.copy(c_l0[:64, :208], D[m_base + 0 : m_base + 64, n_base + 0 : n_base + 208])
                    T.ascend_pipe_barrier("PIPE_ALL")
                elif n_tile == 34:
                    T.copy(
                        B[n_base : n_base + 96, 0:128],
                        b_l1[0, :96, :],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.copy(A[m_base : m_base + 64, 0:128], a_l1[0, :64, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.copy(
                        B[n_base : n_base + 96, 128:256],
                        b_l1[1, :96, :],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(b_l1[0, 0:96, 0:64], b_l0[0, :96, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.copy(b_l1[0, 0:96, 64:128], b_l0[1, :96, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:64, 0:64], a_l0[0, :64, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.copy(a_l1[0, 0:64, 64:128], a_l0[1, :64, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[0, :64, :],
                        b_l0[0, :96, :],
                        c_l0[:64, :96],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 2)
                    for cycle_0 in range(6):
                        T.copy(
                            A[m_base : m_base + 64, 128 + 256 * cycle_0 : 256 + 256 * cycle_0],
                            a_l1[1, :64, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 1)
                        T.ascend_wait_flag("MTE1_M", 1)
                        T.ascend_wait_flag("MTE1_M", 3)
                        T.gemm(
                            a_l0[1, :64, :],
                            b_l0[1, :96, :],
                            c_l0[:64, :96],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 1)
                        T.ascend_set_flag("M_MTE1", 3)
                        T.ascend_wait_flag("MTE1_MTE2", 2)
                        T.copy(
                            B[n_base : n_base + 96, 256 + 256 * cycle_0 : 384 + 256 * cycle_0],
                            b_l1[0, :96, :],
                            l2_cache_ctrl="NOTALLOC_KEEP",
                        )
                        T.ascend_set_flag("MTE2_MTE1", 2)
                        T.ascend_wait_flag("M_MTE1", 2)
                        T.ascend_wait_flag("MTE2_MTE1", 3)
                        T.copy(b_l1[1, 0:96, 0:64], b_l0[0, :96, :])
                        T.ascend_set_flag("MTE1_M", 2)
                        T.ascend_wait_flag("M_MTE1", 3)
                        T.copy(b_l1[1, 0:96, 64:128], b_l0[1, :96, :])
                        T.ascend_set_flag("MTE1_M", 3)
                        T.ascend_set_flag("MTE1_MTE2", 3)
                        T.ascend_wait_flag("M_MTE1", 0)
                        T.ascend_wait_flag("MTE2_MTE1", 1)
                        T.copy(a_l1[1, 0:64, 0:64], a_l0[0, :64, :])
                        T.ascend_set_flag("MTE1_M", 0)
                        T.ascend_wait_flag("M_MTE1", 1)
                        T.copy(a_l1[1, 0:64, 64:128], a_l0[1, :64, :])
                        T.ascend_set_flag("MTE1_M", 1)
                        T.ascend_set_flag("MTE1_MTE2", 1)
                        T.ascend_wait_flag("MTE1_M", 0)
                        T.ascend_wait_flag("MTE1_M", 2)
                        T.gemm(
                            a_l0[0, :64, :],
                            b_l0[0, :96, :],
                            c_l0[:64, :96],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 0)
                        T.ascend_set_flag("M_MTE1", 2)
                        T.ascend_wait_flag("MTE1_MTE2", 0)
                        T.copy(
                            A[m_base : m_base + 64, 256 + 256 * cycle_0 : 384 + 256 * cycle_0],
                            a_l1[0, :64, :],
                        )
                        T.ascend_set_flag("MTE2_MTE1", 0)
                        T.ascend_wait_flag("MTE1_M", 1)
                        T.ascend_wait_flag("MTE1_M", 3)
                        T.gemm(
                            a_l0[1, :64, :],
                            b_l0[1, :96, :],
                            c_l0[:64, :96],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 1)
                        T.ascend_set_flag("M_MTE1", 3)
                        T.ascend_wait_flag("MTE1_MTE2", 3)
                        T.copy(
                            B[n_base : n_base + 96, 384 + 256 * cycle_0 : 512 + 256 * cycle_0],
                            b_l1[1, :96, :],
                            l2_cache_ctrl="NOTALLOC_KEEP",
                        )
                        T.ascend_set_flag("MTE2_MTE1", 3)
                        T.ascend_wait_flag("M_MTE1", 2)
                        T.ascend_wait_flag("MTE2_MTE1", 2)
                        T.copy(b_l1[0, 0:96, 0:64], b_l0[0, :96, :])
                        T.ascend_set_flag("MTE1_M", 2)
                        T.ascend_wait_flag("M_MTE1", 3)
                        T.copy(b_l1[0, 0:96, 64:128], b_l0[1, :96, :])
                        T.ascend_set_flag("MTE1_M", 3)
                        T.ascend_set_flag("MTE1_MTE2", 2)
                        T.ascend_wait_flag("M_MTE1", 0)
                        T.ascend_wait_flag("MTE2_MTE1", 0)
                        T.copy(a_l1[0, 0:64, 0:64], a_l0[0, :64, :])
                        T.ascend_set_flag("MTE1_M", 0)
                        T.ascend_wait_flag("M_MTE1", 1)
                        T.copy(a_l1[0, 0:64, 64:128], a_l0[1, :64, :])
                        T.ascend_set_flag("MTE1_M", 1)
                        T.ascend_set_flag("MTE1_MTE2", 0)
                        T.ascend_wait_flag("MTE1_M", 0)
                        T.ascend_wait_flag("MTE1_M", 2)
                        T.gemm(
                            a_l0[0, :64, :],
                            b_l0[0, :96, :],
                            c_l0[:64, :96],
                            transpose_B=True,
                            clear_accum=False,
                        )
                        T.ascend_set_flag("M_MTE1", 0)
                        T.ascend_set_flag("M_MTE1", 2)
                        T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 64, 1664:1792], a_l1[1, :64, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :64, :],
                        b_l0[1, :96, :],
                        c_l0[:64, :96],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(
                        B[n_base : n_base + 96, 1792:1920],
                        b_l1[0, :96, :],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(b_l1[1, 0:96, 0:64], b_l0[0, :96, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.copy(b_l1[1, 0:96, 64:128], b_l0[1, :96, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:64, 0:64], a_l0[0, :64, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 0:64, 64:128], a_l0[1, :64, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[0, :64, :],
                        b_l0[0, :96, :],
                        c_l0[:64, :96],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 64, 1792:1920], a_l1[0, :64, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :64, :],
                        b_l0[1, :96, :],
                        c_l0[:64, :96],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(
                        B[n_base : n_base + 96, 1920:2048],
                        b_l1[1, :96, :],
                        l2_cache_ctrl="NOTALLOC_KEEP",
                    )
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(b_l1[0, 0:96, 0:64], b_l0[0, :96, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.copy(b_l1[0, 0:96, 64:128], b_l0[1, :96, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:64, 0:64], a_l0[0, :64, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 0:64, 64:128], a_l0[1, :64, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[0, :64, :],
                        b_l0[0, :96, :],
                        c_l0[:64, :96],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 64, 1920:2048], a_l1[1, :64, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :64, :],
                        b_l0[1, :96, :],
                        c_l0[:64, :96],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(b_l1[1, 0:96, 0:64], b_l0[0, :96, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.copy(b_l1[1, 0:96, 64:128], b_l0[1, :96, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:64, 0:64], a_l0[0, :64, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 0:64, 64:128], a_l0[1, :64, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[0, :64, :],
                        b_l0[0, :96, :],
                        c_l0[:64, :96],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :64, :],
                        b_l0[1, :96, :],
                        c_l0[:64, :96],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_FIX", 0)
                    T.ascend_wait_flag("M_FIX", 0)
                    T.copy(c_l0[:64, :96], D[m_base + 0 : m_base + 64, n_base + 0 : n_base + 96])
                    T.ascend_pipe_barrier("PIPE_ALL")


def compile_kernel():
    return tilelang.compile(
        main,
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_auto_schedule": False, "tl.disable_shared_memory_reuse": True},
    )
