import tilelang
import tilelang.ascend.language as T


@T.macro
def swizzle(tile_idx):
    m_tile = T.alloc_var("int32")
    n_tile = T.alloc_var("int32")
    if tile_idx >= 144:
        m_tile = tile_idx // 9
        n_tile = tile_idx % 9
    else:
        superrow = tile_idx // 36
        within = tile_idx % 36
        if within < 32:
            group = within // 32
            local = within % 32
            m_tile = superrow * 4 + local % 4
            n_group = T.Select(superrow % 2 == 0, group, 0 - group)
            n_tile = n_group * 8 + ((local // 4) ^ (local % 4))
        else:
            remainder = within - 32
            m_tile = superrow * 4 + remainder % 4
            n_tile = 8 + remainder // 4
    return m_tile, n_tile


@T.prim_func
def main(
    A: T.Tensor((4096, 7168), "bfloat16"),
    B: T.Tensor((2112, 7168), "bfloat16"),
    D: T.Tensor((4096, 2112), "bfloat16"),
):
    with T.Kernel(36) as bx:  # noqa: SIM117 - Keep the measured TileLang scope structure.
        with T.Cube():
            a_l0 = T.alloc_l0a((3, 128, 64), "bfloat16")
            a_l1 = T.alloc_l1((4, 256, 128), "bfloat16")
            b_l0 = T.alloc_l0b((2, 240, 64), "bfloat16")
            b_l1 = T.alloc_l1((4, 240, 128), "bfloat16")
            c_l0_0_0 = T.alloc_l0c((128, 240), "float32")
            c_l0_1_0 = T.alloc_l0c((128, 240), "float32")
            for tile_idx in T.Persistent([144], 36, bx):
                m_tile, n_tile = swizzle(tile_idx)
                T.assume(m_tile >= 0)
                T.assume(m_tile < 16)
                T.assume(n_tile >= 0)
                T.assume(n_tile < 9)
                m_base = m_tile * 256
                n_base = n_tile * 240
                if n_tile < 8:
                    T.copy(B[n_base : n_base + 240, 0:128], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.copy(A[m_base : m_base + 256, 0:128], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.copy(B[n_base : n_base + 240, 128:256], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.copy(A[m_base : m_base + 256, 128:256], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.copy(B[n_base : n_base + 240, 256:384], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.copy(A[m_base : m_base + 256, 256:384], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.copy(B[n_base : n_base + 240, 384:512], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.copy(A[m_base : m_base + 256, 384:512], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 240, 512:640], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 512:640], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 240, 640:768], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 640:768], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 240, 768:896], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 768:896], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 240, 896:1024], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 896:1024], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 240, 1024:1152], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 1024:1152], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 240, 1152:1280], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 1152:1280], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 240, 1280:1408], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 1280:1408], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 240, 1408:1536], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 1408:1536], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 240, 1536:1664], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 1536:1664], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 240, 1664:1792], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 1664:1792], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 240, 1792:1920], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 1792:1920], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 240, 1920:2048], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 1920:2048], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 240, 2048:2176], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 2048:2176], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 240, 2176:2304], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 2176:2304], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 240, 2304:2432], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 2304:2432], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 240, 2432:2560], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 2432:2560], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 240, 2560:2688], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 2560:2688], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 240, 2688:2816], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 2688:2816], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 240, 2816:2944], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 2816:2944], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 240, 2944:3072], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 2944:3072], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 240, 3072:3200], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 3072:3200], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 240, 3200:3328], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 3200:3328], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 240, 3328:3456], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 3328:3456], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 240, 3456:3584], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 3456:3584], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 240, 3584:3712], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 3584:3712], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 240, 3712:3840], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 3712:3840], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 240, 3840:3968], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 3840:3968], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 240, 3968:4096], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 3968:4096], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 240, 4096:4224], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 4096:4224], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 240, 4224:4352], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 4224:4352], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 240, 4352:4480], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 4352:4480], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 240, 4480:4608], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 4480:4608], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 240, 4608:4736], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 4608:4736], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 240, 4736:4864], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 4736:4864], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 240, 4864:4992], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 4864:4992], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 240, 4992:5120], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 4992:5120], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 240, 5120:5248], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 5120:5248], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 240, 5248:5376], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 5248:5376], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 240, 5376:5504], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 5376:5504], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 240, 5504:5632], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 5504:5632], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 240, 5632:5760], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 5632:5760], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 240, 5760:5888], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 5760:5888], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 240, 5888:6016], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 5888:6016], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 240, 6016:6144], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 6016:6144], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 240, 6144:6272], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 6144:6272], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 240, 6272:6400], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 6272:6400], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 240, 6400:6528], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 6400:6528], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 240, 6528:6656], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 6528:6656], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 240, 6656:6784], b_l1[0, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 6656:6784], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 240, 6784:6912], b_l1[1, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 6784:6912], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 240, 6912:7040], b_l1[2, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 6912:7040], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 240, 7040:7168], b_l1[3, :240, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 7040:7168], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:240, 0:64], b_l0[0, :240, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:240, 64:128], b_l0[1, :240, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :240, :],
                        c_l0_0_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_FIX", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :240, :],
                        c_l0_1_0[:128, :240],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_FIX", 1)
                    T.ascend_wait_flag("M_FIX", 0)
                    T.copy(
                        c_l0_0_0[:128, :240],
                        D[m_base + 0 : m_base + 128, n_base + 0 : n_base + 240],
                    )
                    T.ascend_wait_flag("M_FIX", 1)
                    T.copy(
                        c_l0_1_0[:128, :240],
                        D[m_base + 128 : m_base + 256, n_base + 0 : n_base + 240],
                    )
                    T.ascend_pipe_barrier("PIPE_ALL")
                elif n_tile == 8:
                    T.copy(B[n_base : n_base + 192, 0:128], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.copy(A[m_base : m_base + 256, 0:128], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.copy(B[n_base : n_base + 192, 128:256], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 128:256], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.copy(B[n_base : n_base + 192, 256:384], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 256:384], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.copy(B[n_base : n_base + 192, 384:512], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 384:512], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 192, 512:640], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 512:640], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 192, 640:768], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 640:768], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 192, 768:896], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 768:896], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 192, 896:1024], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 896:1024], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 192, 1024:1152], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 1024:1152], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 192, 1152:1280], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 1152:1280], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 192, 1280:1408], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 1280:1408], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 192, 1408:1536], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 1408:1536], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 192, 1536:1664], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 1536:1664], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 192, 1664:1792], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 1664:1792], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 192, 1792:1920], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 1792:1920], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 192, 1920:2048], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 1920:2048], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 192, 2048:2176], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 2048:2176], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 192, 2176:2304], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 2176:2304], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 192, 2304:2432], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 2304:2432], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 192, 2432:2560], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 2432:2560], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 192, 2560:2688], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 2560:2688], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 192, 2688:2816], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 2688:2816], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 192, 2816:2944], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 2816:2944], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 192, 2944:3072], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 2944:3072], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 192, 3072:3200], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 3072:3200], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 192, 3200:3328], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 3200:3328], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 192, 3328:3456], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 3328:3456], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 192, 3456:3584], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 3456:3584], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 192, 3584:3712], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 3584:3712], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 192, 3712:3840], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 3712:3840], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 192, 3840:3968], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 3840:3968], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 192, 3968:4096], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 3968:4096], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 192, 4096:4224], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 4096:4224], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 192, 4224:4352], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 4224:4352], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 192, 4352:4480], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 4352:4480], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 192, 4480:4608], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 4480:4608], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 192, 4608:4736], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 4608:4736], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 192, 4736:4864], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 4736:4864], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 192, 4864:4992], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 4864:4992], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 192, 4992:5120], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 4992:5120], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 192, 5120:5248], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 5120:5248], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 192, 5248:5376], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 5248:5376], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 192, 5376:5504], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 5376:5504], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 192, 5504:5632], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 5504:5632], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 192, 5632:5760], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 5632:5760], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 192, 5760:5888], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 5760:5888], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 192, 5888:6016], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 5888:6016], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 192, 6016:6144], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 6016:6144], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 192, 6144:6272], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 6144:6272], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 192, 6272:6400], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 6272:6400], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 192, 6400:6528], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 6400:6528], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 6)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 192, 6528:6656], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 6528:6656], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_set_flag("MTE1_MTE2", 7)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                    T.copy(B[n_base : n_base + 192, 6656:6784], b_l1[0, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[m_base : m_base + 256, 6656:6784], a_l1[0, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    T.copy(a_l1[0, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[0, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[0, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                    T.copy(B[n_base : n_base + 192, 6784:6912], b_l1[1, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 5)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[0, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[0, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                    T.copy(A[m_base : m_base + 256, 6784:6912], a_l1[1, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 5)
                    T.copy(b_l1[1, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.ascend_wait_flag("MTE2_MTE1", 1)
                    T.copy(a_l1[1, 0:128, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[1, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[1, 128:256, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 6)
                    T.copy(B[n_base : n_base + 192, 6912:7040], b_l1[2, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 6)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[1, 0:128, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[1, 128:256, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                    T.copy(A[m_base : m_base + 256, 6912:7040], a_l1[2, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                    T.copy(b_l1[2, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.ascend_wait_flag("MTE2_MTE1", 2)
                    T.copy(a_l1[2, 0:128, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[2, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[2, 128:256, 0:64], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_MTE2", 7)
                    T.copy(B[n_base : n_base + 192, 7040:7168], b_l1[3, :192, :])
                    T.ascend_set_flag("MTE2_MTE1", 7)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[2, 0:128, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 2)
                    T.ascend_set_flag("M_MTE1", 3)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[2, 128:256, 64:128], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                    T.copy(A[m_base : m_base + 256, 7040:7168], a_l1[3, :256, :])
                    T.ascend_set_flag("MTE2_MTE1", 3)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 0)
                    T.ascend_set_flag("M_MTE1", 4)
                    T.ascend_wait_flag("M_MTE1", 3)
                    T.ascend_wait_flag("MTE2_MTE1", 7)
                    T.copy(b_l1[3, 0:192, 0:64], b_l0[0, :192, :])
                    T.ascend_set_flag("MTE1_M", 3)
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 3)
                    T.copy(a_l1[3, 0:128, 0:64], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 3)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[0, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_MTE1", 1)
                    T.ascend_wait_flag("M_MTE1", 4)
                    T.copy(b_l1[3, 0:192, 64:128], b_l0[1, :192, :])
                    T.ascend_set_flag("MTE1_M", 4)
                    T.ascend_wait_flag("M_MTE1", 0)
                    T.copy(a_l1[3, 128:256, 0:64], a_l0[0, :128, :])
                    T.ascend_set_flag("MTE1_M", 0)
                    T.ascend_wait_flag("M_MTE1", 2)
                    T.copy(a_l1[3, 0:128, 64:128], a_l0[2, :128, :])
                    T.ascend_set_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 0)
                    T.gemm(
                        a_l0[0, :128, :],
                        b_l0[0, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_wait_flag("M_MTE1", 1)
                    T.copy(a_l1[3, 128:256, 64:128], a_l0[1, :128, :])
                    T.ascend_set_flag("MTE1_M", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                    T.ascend_wait_flag("MTE1_M", 4)
                    T.gemm(
                        a_l0[2, :128, :],
                        b_l0[1, :192, :],
                        c_l0_0_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_FIX", 0)
                    T.ascend_wait_flag("MTE1_M", 1)
                    T.gemm(
                        a_l0[1, :128, :],
                        b_l0[1, :192, :],
                        c_l0_1_0[:128, :192],
                        transpose_B=True,
                        clear_accum=False,
                    )
                    T.ascend_set_flag("M_FIX", 1)
                    T.ascend_wait_flag("M_FIX", 0)
                    T.copy(
                        c_l0_0_0[:128, :192],
                        D[m_base + 0 : m_base + 128, n_base + 0 : n_base + 192],
                    )
                    T.ascend_wait_flag("M_FIX", 1)
                    T.copy(
                        c_l0_1_0[:128, :192],
                        D[m_base + 128 : m_base + 256, n_base + 0 : n_base + 192],
                    )
                    T.ascend_pipe_barrier("PIPE_ALL")


def compile_kernel():
    return tilelang.compile(
        main,
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_auto_schedule": False, "tl.disable_shared_memory_reuse": True},
    )
