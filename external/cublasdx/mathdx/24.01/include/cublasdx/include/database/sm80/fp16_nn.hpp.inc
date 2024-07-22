template<>
struct generated_config<__half, __half, 16, 16, 16, NN, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>
                                                            > {};
template<>
struct generated_config<__half, __half, 32, 32, 32, NN, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 32, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>
                                                            > {};
template<>
struct generated_config<__half, __half, 48, 48, 48, NN, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 8>
                                                            > {};
template<>
struct generated_config<__half, __half, 64, 64, 64, NN, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<__half, __half, 80, 80, 80, NN, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<__half, __half, 96, 96, 96, NN, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<__half, __half, 112, 112, 112, NN, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<__half, __half, 128, 128, 128, NN, SM<800>> : generated_config_list<
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
template<>
struct generated_config<__half, __half, 144, 144, 144, NN, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 16, 2>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<__half, __half, 160, 160, 160, NN, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
