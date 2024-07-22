template<>
struct generated_config<double, double, 16, 16, 16, NN, SM<700>> : generated_config_list<
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 8, 128>,
    generated_config_impl<512, mma_atom::universal_fma, 4, 128>
                                                            > {};
template<>
struct generated_config<double, double, 32, 32, 32, NN, SM<700>> : generated_config_list<
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 32, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 32, 2>,
    generated_config_impl<1024, mma_atom::universal_fma, 8, 128>
                                                            > {};
template<>
struct generated_config<double, double, 48, 48, 48, NN, SM<700>> : generated_config_list<
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>
                                                            > {};
template<>
struct generated_config<double, double, 64, 64, 64, NN, SM<700>> : generated_config_list<
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
