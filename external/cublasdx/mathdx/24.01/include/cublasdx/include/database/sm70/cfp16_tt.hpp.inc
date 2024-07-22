template<>
struct generated_config<complex<__half>, complex<__half>, 16, 16, 16, TT, SM<700>> : generated_config_list<
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 32, 32, 32, TT, SM<700>> : generated_config_list<
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 8, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 48, 48, 48, TT, SM<700>> : generated_config_list<
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 64, 64, 64, TT, SM<700>> : generated_config_list<
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 8, 64>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 80, 80, 80, TT, SM<700>> : generated_config_list<
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>
                                                            > {};
