template<>
struct generated_config<complex<__half>, complex<__half>, 16, 16, 16, NN, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 32, 32, 32, NN, SM<800>> : generated_config_list<
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 32, 4>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 48, 48, 48, NN, SM<800>> : generated_config_list<
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 64, 8>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 64, 64, 64, NN, SM<800>> : generated_config_list<
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 32, 8>,
    generated_config_impl<128, mma_atom::universal_fma, 32, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 80, 80, 80, NN, SM<800>> : generated_config_list<
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 96, 96, 96, NN, SM<800>> : generated_config_list<
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 32, 8>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 112, 112, 112, NN, SM<800>> : generated_config_list<
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
