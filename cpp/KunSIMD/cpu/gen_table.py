import math
table = [1.0, 0.96875, 0.9375, 0.90625, 0.875, 0.84375, 0.84375, 0.8125,
0.78125, 0.78125, 0.75, 0.75, 0.71875, 0.71875, 0.6875,
0.6875, 1.3125, 1.3125, 1.25, 1.25, 1.25, 1.1875, 1.1875, 1.125, 1.125, 1.125, 1.125,
1.0625, 1.0625, 1.0625, 1.0, 1.0]

assert(len(table)==32)
fp32_shift = -math.log(2) * 127
fp32_table = [str(-math.log(x) + fp32_shift)+"f" for x in table]
fp32_values = [str(x) + "f" for x in table]
fp32_templ = f'''
template<>
alignas(64) const float LogLookupTable<float>::r_table[32] = {{{", ".join(fp32_values)}}};
template<>
alignas(64) const float LogLookupTable<float>::logr_table[32] = {{{", ".join(fp32_table)}}};
'''
print(fp32_templ)

fp64_shift = -math.log(2) * 1023
fp64_table = [str(-math.log(x) + fp64_shift) for x in table]
fp64_values = [str(x) for x in table]
fp64_templ = f'''
template<>
alignas(64) const double LogLookupTable<double>::r_table[32] = {{{", ".join(fp64_values)}}};
template<>
alignas(64) const double LogLookupTable<double>::logr_table[32] = {{{", ".join(fp64_table)}}};
'''
print(fp64_templ)