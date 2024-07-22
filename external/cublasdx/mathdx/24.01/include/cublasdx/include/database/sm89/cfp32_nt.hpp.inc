template<>
struct generated_config<complex<float>, complex<float>, 64, 64, 64, NT, SM<890>> : generated_config_list<
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<complex<float>, complex<float>, 32, 32, 32, NT, SM<890>> : generated_config_list<
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>
                                                            > {};
template<>
struct generated_config<complex<float>, complex<float>, 16, 16, 16, NT, SM<890>> : generated_config_list<
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<complex<float>, complex<float>, 48, 48, 48, NT, SM<890>> : generated_config_list<
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
