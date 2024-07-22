template<>
struct generated_config<complex<__half>, complex<__half>, 16, 16, 16, TN, SM<900>> : generated_config_list<
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 32, 32, 32, TN, SM<900>> : generated_config_list<
    generated_config_impl<256, mma_atom::universal_fma, 8, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<64, mma_atom::universal_fma, 32, 2>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 48, 48, 48, TN, SM<900>> : generated_config_list<
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 128, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 64, 64, 64, TN, SM<900>> : generated_config_list<
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 80, 80, 80, TN, SM<900>> : generated_config_list<
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 96, 96, 96, TN, SM<900>> : generated_config_list<
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 112, 112, 112, TN, SM<900>> : generated_config_list<
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 128, 128, 128, TN, SM<900>> : generated_config_list<
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<complex<__half>, complex<__half>, 136, 136, 136, TN, SM<900>> : generated_config_list<
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
