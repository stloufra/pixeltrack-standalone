template<>
struct generated_config<__half, __half, 16, 16, 16, TT, SM<700>> : generated_config_list<
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>
                                                            > {};
template<>
struct generated_config<__half, __half, 32, 32, 32, TT, SM<700>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<256, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 8>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 16>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 64, 2>
                                                            > {};
template<>
struct generated_config<__half, __half, 48, 48, 48, TT, SM<700>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 2, 8>,
    generated_config_impl<256, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 2, 16>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 2, 32>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 64, 2>
                                                            > {};
template<>
struct generated_config<__half, __half, 64, 64, 64, TT, SM<700>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<256, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 8>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 8, 8>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 8, 16>
                                                            > {};
template<>
struct generated_config<__half, __half, 80, 80, 80, TT, SM<700>> : generated_config_list<
    generated_config_impl<256, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 2, 16>,
    generated_config_impl<128, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 8, 8>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 8, 16>
                                                            > {};
template<>
struct generated_config<__half, __half, 96, 96, 96, TT, SM<700>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<256, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 8>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 16>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 32>
                                                            > {};
template<>
struct generated_config<__half, __half, 112, 112, 112, TT, SM<700>> : generated_config_list<
    generated_config_impl<256, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 2, 16>,
    generated_config_impl<128, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 16>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 32, 4>
                                                            > {};
template<>
struct generated_config<__half, __half, 128, 128, 128, TT, SM<700>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<256, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 16, 4>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F16F16F16F16_TN_CUBLASDX, 16, 8>
                                                            > {};
