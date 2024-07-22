template<>
struct generated_config<__half, __half, 16, 16, 16, NT, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<__half, __half, 32, 32, 32, NT, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>
                                                            > {};
template<>
struct generated_config<__half, __half, 48, 48, 48, NT, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 8>
                                                            > {};
template<>
struct generated_config<__half, __half, 64, 64, 64, NT, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<__half, __half, 80, 80, 80, NT, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<__half, __half, 96, 96, 96, NT, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<__half, __half, 112, 112, 112, NT, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<__half, __half, 128, 128, 128, NT, SM<800>> : generated_config_list<
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
template<>
struct generated_config<__half, __half, 144, 144, 144, NT, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<__half, __half, 160, 160, 160, NT, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
