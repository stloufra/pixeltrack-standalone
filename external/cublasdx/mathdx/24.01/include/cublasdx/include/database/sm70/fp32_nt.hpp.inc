template<>
struct generated_config<float, float, 16, 16, 16, NT, SM<700>> : generated_config_list<
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 8, 128>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>
                                                            > {};
template<>
struct generated_config<float, float, 32, 32, 32, NT, SM<700>> : generated_config_list<
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<float, float, 48, 48, 48, NT, SM<700>> : generated_config_list<
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<float, float, 64, 64, 64, NT, SM<700>> : generated_config_list<
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<float, float, 80, 80, 80, NT, SM<700>> : generated_config_list<
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
