import tilelang
import tilelang.ascend.language as T


@T.macro
def swizzle(tile_idx):
    m_tile = T.alloc_var("int32")
    n_tile = T.alloc_var("int32")
    if tile_idx >= 1536:
        m_tile = tile_idx // 96
        n_tile = tile_idx % 96
    else:
        superrow = tile_idx // 384
        within = tile_idx % 384
        if within < 384:
            group = within // 32
            local = within % 32
            m_tile = superrow * 4 + local % 4
            n_group = T.Select(superrow % 2 == 0, group, 11 - group)
            n_tile = n_group * 8 + ((local // 4) ^ (local % 4))
        else:
            remainder = within - 384
            m_tile = superrow * 4 + remainder % 4
            n_tile = 96 + remainder // 4
    return m_tile, n_tile


@T.prim_func
def main(
    A: T.Tensor((4096, 1536), "bfloat16"),
    B: T.Tensor((24576, 1536), "bfloat16"),
    D: T.Tensor((4096, 24576), "bfloat16"),
):
    with T.Kernel(36) as bx:  # noqa: SIM117 - Keep the measured TileLang scope structure.
        with T.Cube():
            a_l0 = T.alloc_l0a((2, 256, 64), "bfloat16")
            a_l1 = T.alloc_l1((4, 256, 128), "bfloat16")
            b_l0 = T.alloc_l0b((3, 128, 64), "bfloat16")
            b_l1 = T.alloc_l1((4, 256, 128), "bfloat16")
            c_l0_0_0 = T.alloc_l0c((256, 128), "float32")
            c_l0_0_1 = T.alloc_l0c((256, 128), "float32")
            for tile_idx in T.Persistent([1536], 36, bx):
                m_tile, n_tile = swizzle(tile_idx)
                T.assume(m_tile >= 0)
                T.assume(m_tile < 16)
                T.assume(n_tile >= 0)
                T.assume(n_tile < 96)
                m_base = m_tile * 256
                n_base = n_tile * 256
                T.copy(A[m_base : m_base + 256, 0:128], a_l1[0, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 0)
                T.copy(B[n_base : n_base + 256, 0:128], b_l1[0, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 4)
                T.ascend_wait_flag("MTE2_MTE1", 0)
                T.copy(a_l1[0, 0:256, 0:64], a_l0[0, :256, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE2_MTE1", 4)
                T.copy(b_l1[0, 0:128, 0:64], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=True,
                )
                T.ascend_set_flag("M_MTE1", 2)
                T.copy(a_l1[0, 0:256, 64:128], a_l0[1, :256, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_set_flag("MTE1_MTE2", 0)
                T.copy(b_l1[0, 128:256, 0:64], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.copy(A[m_base : m_base + 256, 128:256], a_l1[1, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 1)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=True,
                )
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 4)
                T.copy(b_l1[0, 0:128, 64:128], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[0, 128:256, 64:128], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_set_flag("MTE1_MTE2", 4)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 3)
                T.copy(B[n_base : n_base + 256, 128:256], b_l1[1, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 5)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 1)
                T.copy(a_l1[1, 0:256, 0:64], a_l0[0, :256, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 3)
                T.ascend_wait_flag("MTE2_MTE1", 5)
                T.copy(b_l1[1, 0:128, 0:64], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("M_MTE1", 1)
                T.copy(a_l1[1, 0:256, 64:128], a_l0[1, :256, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_set_flag("MTE1_MTE2", 1)
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[1, 128:256, 0:64], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.copy(A[m_base : m_base + 256, 256:384], a_l1[2, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 2)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 4)
                T.copy(b_l1[1, 0:128, 64:128], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[1, 128:256, 64:128], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_set_flag("MTE1_MTE2", 5)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 4)
                T.copy(B[n_base : n_base + 256, 256:384], b_l1[2, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 6)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 2)
                T.copy(a_l1[2, 0:256, 0:64], a_l0[0, :256, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("M_MTE1", 4)
                T.ascend_wait_flag("MTE2_MTE1", 6)
                T.copy(b_l1[2, 0:128, 0:64], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 4)
                T.ascend_wait_flag("M_MTE1", 1)
                T.copy(a_l1[2, 0:256, 64:128], a_l0[1, :256, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_set_flag("MTE1_MTE2", 2)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[2, 128:256, 0:64], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.copy(A[m_base : m_base + 256, 384:512], a_l1[3, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 3)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[2, 0:128, 64:128], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("M_MTE1", 4)
                T.copy(b_l1[2, 128:256, 64:128], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_set_flag("MTE1_MTE2", 6)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 2)
                T.copy(B[n_base : n_base + 256, 384:512], b_l1[3, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 7)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 3)
                T.copy(a_l1[3, 0:256, 0:64], a_l0[0, :256, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 4)
                T.ascend_wait_flag("M_MTE1", 2)
                T.ascend_wait_flag("MTE2_MTE1", 7)
                T.copy(b_l1[3, 0:128, 0:64], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 1)
                T.copy(a_l1[3, 0:256, 64:128], a_l0[1, :256, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_set_flag("MTE1_MTE2", 3)
                T.ascend_wait_flag("M_MTE1", 4)
                T.copy(b_l1[3, 128:256, 0:64], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_wait_flag("MTE1_MTE2", 0)
                T.copy(A[m_base : m_base + 256, 512:640], a_l1[0, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 0)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 4)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[3, 0:128, 64:128], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[3, 128:256, 64:128], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_set_flag("MTE1_MTE2", 7)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("MTE1_MTE2", 4)
                T.copy(B[n_base : n_base + 256, 512:640], b_l1[0, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 4)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 0)
                T.copy(a_l1[0, 0:256, 0:64], a_l0[0, :256, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 3)
                T.ascend_wait_flag("MTE2_MTE1", 4)
                T.copy(b_l1[0, 0:128, 0:64], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("M_MTE1", 1)
                T.copy(a_l1[0, 0:256, 64:128], a_l0[1, :256, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_set_flag("MTE1_MTE2", 0)
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[0, 128:256, 0:64], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("MTE1_MTE2", 1)
                T.copy(A[m_base : m_base + 256, 640:768], a_l1[1, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 1)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 4)
                T.copy(b_l1[0, 0:128, 64:128], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[0, 128:256, 64:128], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_set_flag("MTE1_MTE2", 4)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 4)
                T.ascend_wait_flag("MTE1_MTE2", 5)
                T.copy(B[n_base : n_base + 256, 640:768], b_l1[1, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 5)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 1)
                T.copy(a_l1[1, 0:256, 0:64], a_l0[0, :256, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("M_MTE1", 4)
                T.ascend_wait_flag("MTE2_MTE1", 5)
                T.copy(b_l1[1, 0:128, 0:64], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 4)
                T.ascend_wait_flag("M_MTE1", 1)
                T.copy(a_l1[1, 0:256, 64:128], a_l0[1, :256, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_set_flag("MTE1_MTE2", 1)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[1, 128:256, 0:64], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_wait_flag("MTE1_MTE2", 2)
                T.copy(A[m_base : m_base + 256, 768:896], a_l1[2, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 2)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[1, 0:128, 64:128], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("M_MTE1", 4)
                T.copy(b_l1[1, 128:256, 64:128], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_set_flag("MTE1_MTE2", 5)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("MTE1_MTE2", 6)
                T.copy(B[n_base : n_base + 256, 768:896], b_l1[2, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 6)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 2)
                T.copy(a_l1[2, 0:256, 0:64], a_l0[0, :256, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 4)
                T.ascend_wait_flag("M_MTE1", 2)
                T.ascend_wait_flag("MTE2_MTE1", 6)
                T.copy(b_l1[2, 0:128, 0:64], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 1)
                T.copy(a_l1[2, 0:256, 64:128], a_l0[1, :256, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_set_flag("MTE1_MTE2", 2)
                T.ascend_wait_flag("M_MTE1", 4)
                T.copy(b_l1[2, 128:256, 0:64], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_wait_flag("MTE1_MTE2", 3)
                T.copy(A[m_base : m_base + 256, 896:1024], a_l1[3, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 3)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 4)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[2, 0:128, 64:128], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[2, 128:256, 64:128], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_set_flag("MTE1_MTE2", 6)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("MTE1_MTE2", 7)
                T.copy(B[n_base : n_base + 256, 896:1024], b_l1[3, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 7)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 3)
                T.copy(a_l1[3, 0:256, 0:64], a_l0[0, :256, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 3)
                T.ascend_wait_flag("MTE2_MTE1", 7)
                T.copy(b_l1[3, 0:128, 0:64], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("M_MTE1", 1)
                T.copy(a_l1[3, 0:256, 64:128], a_l0[1, :256, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_set_flag("MTE1_MTE2", 3)
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[3, 128:256, 0:64], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("MTE1_MTE2", 0)
                T.copy(A[m_base : m_base + 256, 1024:1152], a_l1[0, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 4)
                T.copy(b_l1[3, 0:128, 64:128], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[3, 128:256, 64:128], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_set_flag("MTE1_MTE2", 7)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 4)
                T.ascend_wait_flag("MTE1_MTE2", 4)
                T.copy(B[n_base : n_base + 256, 1024:1152], b_l1[0, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 4)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 0)
                T.copy(a_l1[0, 0:256, 0:64], a_l0[0, :256, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("M_MTE1", 4)
                T.ascend_wait_flag("MTE2_MTE1", 4)
                T.copy(b_l1[0, 0:128, 0:64], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 4)
                T.ascend_wait_flag("M_MTE1", 1)
                T.copy(a_l1[0, 0:256, 64:128], a_l0[1, :256, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[0, 128:256, 0:64], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_wait_flag("MTE1_MTE2", 1)
                T.copy(A[m_base : m_base + 256, 1152:1280], a_l1[1, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 1)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[0, 0:128, 64:128], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("M_MTE1", 4)
                T.copy(b_l1[0, 128:256, 64:128], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("MTE1_MTE2", 5)
                T.copy(B[n_base : n_base + 256, 1152:1280], b_l1[1, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 5)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 1)
                T.copy(a_l1[1, 0:256, 0:64], a_l0[0, :256, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 4)
                T.ascend_wait_flag("M_MTE1", 2)
                T.ascend_wait_flag("MTE2_MTE1", 5)
                T.copy(b_l1[1, 0:128, 0:64], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 1)
                T.copy(a_l1[1, 0:256, 64:128], a_l0[1, :256, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_wait_flag("M_MTE1", 4)
                T.copy(b_l1[1, 128:256, 0:64], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_wait_flag("MTE1_MTE2", 2)
                T.copy(A[m_base : m_base + 256, 1280:1408], a_l1[2, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 2)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 4)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[1, 0:128, 64:128], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[1, 128:256, 64:128], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("MTE1_MTE2", 6)
                T.copy(B[n_base : n_base + 256, 1280:1408], b_l1[2, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 6)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 2)
                T.copy(a_l1[2, 0:256, 0:64], a_l0[0, :256, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 3)
                T.ascend_wait_flag("MTE2_MTE1", 6)
                T.copy(b_l1[2, 0:128, 0:64], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("M_MTE1", 1)
                T.copy(a_l1[2, 0:256, 64:128], a_l0[1, :256, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[2, 128:256, 0:64], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("MTE1_MTE2", 3)
                T.copy(A[m_base : m_base + 256, 1408:1536], a_l1[3, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 3)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 0)
                T.ascend_set_flag("M_MTE1", 2)
                T.ascend_wait_flag("M_MTE1", 4)
                T.copy(b_l1[2, 0:128, 64:128], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[2, 128:256, 64:128], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 4)
                T.ascend_wait_flag("MTE1_MTE2", 7)
                T.copy(B[n_base : n_base + 256, 1408:1536], b_l1[3, :256, :])
                T.ascend_set_flag("MTE2_MTE1", 7)
                T.ascend_wait_flag("M_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 3)
                T.copy(a_l1[3, 0:256, 0:64], a_l0[0, :256, :])
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 1)
                T.ascend_set_flag("M_MTE1", 3)
                T.ascend_wait_flag("M_MTE1", 4)
                T.ascend_wait_flag("MTE2_MTE1", 7)
                T.copy(b_l1[3, 0:128, 0:64], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_wait_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_MTE1", 4)
                T.ascend_wait_flag("M_MTE1", 1)
                T.copy(a_l1[3, 0:256, 64:128], a_l0[1, :256, :])
                T.ascend_set_flag("MTE1_M", 1)
                T.ascend_wait_flag("M_MTE1", 3)
                T.copy(b_l1[3, 128:256, 0:64], b_l0[1, :128, :])
                T.ascend_set_flag("MTE1_M", 3)
                T.ascend_wait_flag("MTE1_M", 3)
                T.gemm(
                    a_l0[0, :256, :],
                    b_l0[1, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_wait_flag("M_MTE1", 2)
                T.copy(b_l1[3, 0:128, 64:128], b_l0[0, :128, :])
                T.ascend_set_flag("MTE1_M", 2)
                T.ascend_wait_flag("M_MTE1", 4)
                T.copy(b_l1[3, 128:256, 64:128], b_l0[2, :128, :])
                T.ascend_set_flag("MTE1_M", 4)
                T.ascend_wait_flag("MTE1_M", 1)
                T.ascend_wait_flag("MTE1_M", 2)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[0, :128, :],
                    c_l0_0_0[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_FIX", 0)
                T.ascend_wait_flag("MTE1_M", 4)
                T.gemm(
                    a_l0[1, :256, :],
                    b_l0[2, :128, :],
                    c_l0_0_1[:256, :128],
                    transpose_B=True,
                    clear_accum=False,
                )
                T.ascend_set_flag("M_FIX", 1)
                T.ascend_wait_flag("M_FIX", 0)
                T.copy(
                    c_l0_0_0[:256, :128], D[m_base + 0 : m_base + 256, n_base + 0 : n_base + 128]
                )
                T.ascend_wait_flag("M_FIX", 1)
                T.copy(
                    c_l0_0_1[:256, :128], D[m_base + 0 : m_base + 256, n_base + 128 : n_base + 256]
                )
                T.ascend_pipe_barrier("PIPE_ALL")


def compile_kernel():
    return tilelang.compile(
        main,
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_auto_schedule": False, "tl.disable_shared_memory_reuse": True},
    )
