template<>
struct generated_config<complex<double>, complex<double>, 16, 16, 16, NN, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
template<>
struct generated_config<complex<double>, complex<double>, 32, 32, 32, NN, SM<800>> : generated_config_list<
    generated_config_impl<256, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 4, 2>,
    generated_config_impl<128, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 2, 2>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
template<>
struct generated_config<complex<double>, complex<double>, 48, 48, 48, NN, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 2, 8>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
