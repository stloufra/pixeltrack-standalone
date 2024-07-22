template<>
struct generated_config<double, double, 16, 16, 16, NT, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 2>,
    generated_config_impl<1024, mma_atom::universal_fma, 8, 128>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
template<>
struct generated_config<double, double, 32, 32, 32, NT, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
template<>
struct generated_config<double, double, 48, 48, 48, NT, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 8>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
template<>
struct generated_config<double, double, 64, 64, 64, NT, SM<800>> : generated_config_list<
    generated_config_impl<256, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 2>,
    generated_config_impl<128, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 2>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>
                                                            > {};
template<>
struct generated_config<double, double, 80, 80, 80, NT, SM<800>> : generated_config_list<
    generated_config_impl<128, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
                                                            > {};
