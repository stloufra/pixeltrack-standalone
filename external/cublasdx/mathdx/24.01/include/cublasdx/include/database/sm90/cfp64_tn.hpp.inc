template<>
struct generated_config<complex<double>, complex<double>, 16, 16, 16, TN, SM<900>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 4, 2>,
    generated_config_impl<64, mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX, 1, 2>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>
                                                            > {};
template<>
struct generated_config<complex<double>, complex<double>, 32, 32, 32, TN, SM<900>> : generated_config_list<
    generated_config_impl<256, mma_atom::SM90_16x8x16_C64C64C64C64_TN_CUBLASDX, 2, 4>,
    generated_config_impl<128, mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX, 2, 2>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 4, 4>,
    generated_config_impl<64, mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX, 2, 1>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 8, 4>
                                                            > {};
template<>
struct generated_config<complex<double>, complex<double>, 48, 48, 48, TN, SM<900>> : generated_config_list<
    generated_config_impl<256, mma_atom::SM90_16x8x8_C64C64C64C64_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX, 4, 4>,
    generated_config_impl<128, mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX, 2, 2>,
    generated_config_impl<1024, mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX, 4, 8>,
    generated_config_impl<64, mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX, 1, 2>
                                                            > {};
template<>
struct generated_config<complex<double>, complex<double>, 64, 64, 64, TN, SM<900>> : generated_config_list<
    generated_config_impl<256, mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX, 4, 4>,
    generated_config_impl<128, mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX, 2, 2>,
    generated_config_impl<1024, mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX, 4, 8>,
    generated_config_impl<64, mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX, 2, 1>
                                                            > {};
template<>
struct generated_config<complex<double>, complex<double>, 68, 68, 68, TN, SM<900>> : generated_config_list<
    generated_config_impl<256, mma_atom::SM90_16x8x8_C64C64C64C64_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM90_16x8x4_C64C64C64C64_TN_CUBLASDX, 8, 2>,
    generated_config_impl<128, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 2, 2>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_C64C64C64C64_TN_CUBLASDX, 2, 16>,
    generated_config_impl<64, mma_atom::SM90_16x8x16_C64C64C64C64_TN_CUBLASDX, 1, 2>
                                                            > {};
