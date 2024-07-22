template<>
struct generated_config<complex<double>, complex<double>, 16, 16, 16, TN, SM<700>> : generated_config_list<
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
template<>
struct generated_config<complex<double>, complex<double>, 32, 32, 32, TN, SM<700>> : generated_config_list<
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<64, mma_atom::universal_fma, 32, 2>
                                                            > {};
