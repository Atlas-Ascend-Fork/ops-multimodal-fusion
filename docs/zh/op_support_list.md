# 接口支持清单

| 算子名 | Torch API | ascend910b | ascend910_93 | ascend950 |
| ------ | --------- | :--------: | :----------: | :-------: |
| abs    | `torch.ops.ops_multimodal_fusion.abs(Tensor x) -> Tensor` | ✓ | ✓ | ✓ |
| adaptive_avg_pool2d | `torch.ops.ops_multimodal_fusion.adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor` | - | - | ✓ |
| add | `torch.ops.ops_multimodal_fusion.add(Tensor x, Tensor y) -> Tensor` | - | - | ✓ |
| addmv | `torch.ops.ops_multimodal_fusion.addmv(Tensor self, Tensor mat, Tensor vec, Scalar beta=1, Scalar alpha=1) -> Tensor` | - | - | ✓ |
| airy_ai | `torch.ops.ops_multimodal_fusion.airy_ai(Tensor x) -> Tensor` | - | - | ✓ |
| angle | `torch.ops.ops_multimodal_fusion.angle(Tensor input) -> Tensor` | - | - | ✓ |
| any | `torch.ops.ops_multimodal_fusion.any(Tensor x, int dim, bool keepdim) -> Tensor` | - | - | ✓ |
| avg_pool2d | `torch.ops.ops_multimodal_fusion.avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor` | - | - | ✓ |
| bessel_j0 | `torch.ops.ops_multimodal_fusion.bessel_j0(Tensor x) -> Tensor` | - | - | ✓ |
| bessel_j1 | `torch.ops.ops_multimodal_fusion.bessel_j1(Tensor x) -> Tensor` | - | - | ✓ |
| bessel_y0 | `torch.ops.ops_multimodal_fusion.bessel_y0(Tensor x) -> Tensor` | - | - | ✓ |
| bessel_y1 | `torch.ops.ops_multimodal_fusion.bessel_y1(Tensor x) -> Tensor` | - | - | ✓ |
| binomial | `torch.ops.ops_multimodal_fusion.binomial(Tensor count, Tensor prob, int seed=0) -> Tensor` | - | - | ✓ |
| bitwisenot | `torch.ops.ops_multimodal_fusion.bitwisenot(Tensor x) -> Tensor` | - | - | ✓ |
| c2_accuracy | `torch.ops.ops_multimodal_fusion.c2_accuracy(Tensor predictions, Tensor labels, int top_k=1) -> Tensor` | - | - | ✓ |
| c2_affine_channel | `torch.ops.ops_multimodal_fusion.c2_affine_channel(Tensor X, Tensor scale, Tensor bias) -> Tensor` | - | - | ✓ |
| c2_batch_moments | `torch.ops.ops_multimodal_fusion.c2_batch_moments(Tensor X) -> (Tensor mu, Tensor var)` | - | - | ✓ |
| c2_batch_permutation | `torch.ops.ops_multimodal_fusion.c2_batch_permutation(Tensor X, Tensor indices) -> Tensor` | - | - | ✓ |
| c2_boolean_mask | `torch.ops.ops_multimodal_fusion.c2_boolean_mask(Tensor data, Tensor mask) -> (Tensor masked_data, Tensor masked_indices)` | - | - | ✓ |
| c2_boolean_unmask | `torch.ops.ops_multimodal_fusion.c2_boolean_unmask(Tensor[] inputs) -> Tensor` | - | - | ✓ |
| c2_bucketize | `torch.ops.ops_multimodal_fusion.c2_bucketize(Tensor self, float[] boundaries) -> Tensor` | - | - | ✓ |
| c2_cbrt | `torch.ops.ops_multimodal_fusion.c2_cbrt(Tensor self) -> Tensor` | - | - | ✓ |
| c2_lars | `torch.ops.ops_multimodal_fusion.c2_lars(Tensor X, Tensor dX, Tensor wd, Tensor trust, Tensor lr_max, float offset=0.5, float lr_min=0.02) -> Tensor` | - | - | ✓ |
| cauchy | `torch.ops.ops_multimodal_fusion.cauchy(Tensor x, float median=0.0, float sigma=1.0, int seed=0) -> Tensor` | - | - | ✓ |
| cdist_backward | `torch.ops.ops_multimodal_fusion.cdist_backward(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist) -> Tensor` | - | - | ✓ |
| chebyshev_polynomial_t | `torch.ops.ops_multimodal_fusion.chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| chebyshev_polynomial_u | `torch.ops.ops_multimodal_fusion.chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| chebyshev_polynomial_v | `torch.ops.ops_multimodal_fusion.chebyshev_polynomial_v(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| chebyshev_polynomial_w | `torch.ops.ops_multimodal_fusion.chebyshev_polynomial_w(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| complex | `torch.ops.ops_multimodal_fusion.complex(Tensor real, Tensor imag) -> Tensor` | - | - | ✓ |
| conjphysical | `torch.ops.ops_multimodal_fusion.conjphysical(Tensor x) -> Tensor` | - | - | ✓ |
| copysign | `torch.ops.ops_multimodal_fusion.copysign(Tensor a, Tensor b) -> Tensor` | - | - | ✓ |
| cummax | `torch.ops.ops_multimodal_fusion.cummax(Tensor x, int dim) -> (Tensor values, Tensor indices)` | - | - | ✓ |
| cumprod | `torch.ops.ops_multimodal_fusion.cumprod(Tensor x, int dim) -> Tensor` | - | - | ✓ |
| depthwise_conv3d | `torch.ops.ops_multimodal_fusion.depthwise_conv3d(Tensor input, Tensor weight, Tensor? bias, int[3] stride, int[3] padding, int[3] dilation) -> Tensor` | - | - | ✓ |
| dequant_swiglu_quant | `torch.ops.ops_multimodal_fusion.dequant_swiglu_quant(Tensor x, Tensor? weight_scale, Tensor? act_scale) -> (Tensor, Tensor)` | - | - | ✓ |
| digamma | `torch.ops.ops_multimodal_fusion.digamma(Tensor x) -> Tensor` | - | - | ✓ |
| dirichlet | `torch.ops.ops_multimodal_fusion.dirichlet(Tensor alpha, int seed=0) -> Tensor` | - | - | ✓ |
| dynamic_quant | `torch.ops.ops_multimodal_fusion.dynamic_quant(Tensor x, bool symmetric=True) -> (Tensor, Tensor)` | - | - | ✓ |
| entr | `torch.ops.ops_multimodal_fusion.entr(Tensor x) -> Tensor` | - | - | ✓ |
| erfcx | `torch.ops.ops_multimodal_fusion.erfcx(Tensor x) -> Tensor` | - | - | ✓ |
| exp | `torch.ops.ops_multimodal_fusion.exp(Tensor x) -> Tensor` | - | - | ✓ |
| exp2 | `torch.ops.ops_multimodal_fusion.exp2(Tensor x) -> Tensor` | - | - | ✓ |
| exponential | `torch.ops.ops_multimodal_fusion.exponential(Tensor x, float lambd=1.0, int seed=0) -> Tensor` | - | - | ✓ |
| fft_conj_symmetry | `torch.ops.ops_multimodal_fusion.fft_conj_symmetry(Tensor input, int dim, int out_size) -> Tensor` | - | - | ✓ |
| foreach_ceil | `torch.ops.ops_multimodal_fusion.foreach_ceil(Tensor[] tensors) -> Tensor[]` | - | - | ✓ |
| foreach_floor | `torch.ops.ops_multimodal_fusion.foreach_floor(Tensor[] tensors) -> Tensor[]` | - | - | ✓ |
| foreach_frac | `torch.ops.ops_multimodal_fusion.foreach_frac(Tensor[] tensors) -> Tensor[]` | - | - | ✓ |
| foreach_lgamma | `torch.ops.ops_multimodal_fusion.foreach_lgamma(Tensor[] tensors) -> Tensor[]` | - | - | ✓ |
| foreach_trunc | `torch.ops.ops_multimodal_fusion.foreach_trunc(Tensor[] tensors) -> Tensor[]` | - | - | ✓ |
| fractional_max_pool2d | `torch.ops.ops_multimodal_fusion.fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)` | - | - | ✓ |
| fractional_max_pool3d | `torch.ops.ops_multimodal_fusion.fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)` | - | - | ✓ |
| frexp | `torch.ops.ops_multimodal_fusion.frexp(Tensor x) -> (Tensor mantissa, Tensor exponent)` | - | - | ✓ |
| gamma | `torch.ops.ops_multimodal_fusion.gamma(Tensor alpha, int seed=0) -> Tensor` | - | - | ✓ |
| gelu | `torch.ops.ops_multimodal_fusion.gelu(Tensor x) -> Tensor` | - | - | ✓ |
| geometric | `torch.ops.ops_multimodal_fusion.geometric(Tensor x, float p, int seed=0) -> Tensor` | - | - | ✓ |
| get_indice_pairs_subm_lookup | `torch.ops.ops_multimodal_fusion.get_indice_pairs_subm_lookup(Tensor coords, int[] spatial_shape, int[] ksize, int[] dilation, Tensor sorted_enc, Tensor sorted_row) -> Tensor` | - | - | ✓ |
| gru_cell | `torch.ops.ops_multimodal_fusion.gru_cell(Tensor input, Tensor hx, Tensor weight_ih, Tensor weight_hh, Tensor? bias_ih=None, Tensor? bias_hh=None) -> Tensor` | - | - | ✓ |
| hermite_polynomial_h | `torch.ops.ops_multimodal_fusion.hermite_polynomial_h(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| hermite_polynomial_he | `torch.ops.ops_multimodal_fusion.hermite_polynomial_he(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| hypot | `torch.ops.ops_multimodal_fusion.hypot(Tensor x, Tensor y) -> Tensor` | - | - | ✓ |
| igamma | `torch.ops.ops_multimodal_fusion.igamma(Tensor a, Tensor x) -> Tensor` | - | - | ✓ |
| igammac | `torch.ops.ops_multimodal_fusion.igammac(Tensor a, Tensor x) -> Tensor` | - | - | ✓ |
| index_copy | `torch.ops.ops_multimodal_fusion.index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor` | - | - | ✓ |
| index_reduce | `torch.ops.ops_multimodal_fusion.index_reduce(Tensor self, int dim, Tensor index, Tensor source, str reduce, bool include_self=True) -> Tensor` | - | - | ✓ |
| indice_conv | `torch.ops.ops_multimodal_fusion.indice_conv(Tensor gathered, Tensor filters, Tensor k_idx, Tensor s_idx, int P) -> Tensor` | - | - | ✓ |
| int_repr | `torch.ops.ops_multimodal_fusion.int_repr(Tensor x) -> Tensor` | - | - | ✓ |
| kaiserwindow | `torch.ops.ops_multimodal_fusion.kaiserwindow(Tensor x, float beta, int window_length, bool periodic) -> Tensor` | - | - | ✓ |
| kthvalue | `torch.ops.ops_multimodal_fusion.kthvalue(Tensor x, int k, int dim, bool keepdim) -> (Tensor values, Tensor indices)` | - | - | ✓ |
| laguerre_polynomial_l | `torch.ops.ops_multimodal_fusion.laguerre_polynomial_l(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| layer_norm | `torch.ops.ops_multimodal_fusion.layer_norm(Tensor x, Tensor gamma, Tensor beta, float eps=1e-6) -> Tensor` | - | - | ✓ |
| lcm | `torch.ops.ops_multimodal_fusion.lcm(Tensor a, Tensor b) -> Tensor` | - | - | ✓ |
| leftshift | `torch.ops.ops_multimodal_fusion.leftshift(Tensor a, int b) -> Tensor` | - | - | ✓ |
| legendre_polynomial_p | `torch.ops.ops_multimodal_fusion.legendre_polynomial_p(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| log_add_exp2 | `torch.ops.ops_multimodal_fusion.log_add_exp2(Tensor a, Tensor b) -> Tensor` | - | - | ✓ |
| log_normal | `torch.ops.ops_multimodal_fusion.log_normal(Tensor x, float mean=1.0, float std=2.0, int seed=0) -> Tensor` | - | - | ✓ |
| logcumsumexp | `torch.ops.ops_multimodal_fusion.logcumsumexp(Tensor x, int dim) -> Tensor` | - | - | ✓ |
| logicalxor | `torch.ops.ops_multimodal_fusion.logicalxor(Tensor x, Tensor y) -> Tensor` | - | - | ✓ |
| logit | `torch.ops.ops_multimodal_fusion.logit(Tensor x, float? eps=None) -> Tensor` | - | - | ✓ |
| logndtr | `torch.ops.ops_multimodal_fusion.logndtr(Tensor x) -> Tensor` | - | - | ✓ |
| lstm_cell | `torch.ops.ops_multimodal_fusion.lstm_cell(Tensor input, Tensor hx, Tensor cx, Tensor weight_ih, Tensor weight_hh, Tensor? bias_ih=None, Tensor? bias_hh=None) -> (Tensor hy, Tensor cy)` | - | - | ✓ |
| make_per_tensor_quantized | `torch.ops.ops_multimodal_fusion.make_per_tensor_quantized(Tensor x, float scale, int zero_point) -> Tensor` | - | - | ✓ |
| matrix_exp_util | `torch.ops.ops_multimodal_fusion.matrix_exp_util(Tensor input, Tensor coefficients) -> Tensor` | - | - | ✓ |
| max | `torch.ops.ops_multimodal_fusion.max_dim(Tensor x, int dim, bool keepdim) -> (Tensor values, Tensor indices)` | - | - | ✓ |
| max_unpool2d | `torch.ops.ops_multimodal_fusion.max_unpool2d(Tensor input, Tensor indices, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_size) -> Tensor` | - | - | ✓ |
| max_unpool3d | `torch.ops.ops_multimodal_fusion.max_unpool3d(Tensor input, Tensor indices, int[3] kernel_size, int[] stride, int[] padding, int[] output_size) -> Tensor` | - | - | ✓ |
| mean | `torch.ops.ops_multimodal_fusion.mean(Tensor x, int dim, bool keepdim) -> Tensor` | - | - | ✓ |
| mode | `torch.ops.ops_multimodal_fusion.mode(Tensor x, int dim, bool keepdim) -> (Tensor values, Tensor indices)` | - | - | ✓ |
| modified_bessel_k0 | `torch.ops.ops_multimodal_fusion.modified_bessel_k0(Tensor x) -> Tensor` | - | - | ✓ |
| modified_bessel_k1 | `torch.ops.ops_multimodal_fusion.modified_bessel_k1(Tensor x) -> Tensor` | - | - | ✓ |
| mul | `torch.ops.ops_multimodal_fusion.mul(Tensor x, Tensor y) -> Tensor` | - | - | ✓ |
| multi_margin_loss | `torch.ops.ops_multimodal_fusion.multi_margin_loss(Tensor input, Tensor target, Scalar p=1, Scalar margin=1.0, Tensor? weight=None, int reduction=1) -> Tensor` | - | - | ✓ |
| multilabel_margin_loss | `torch.ops.ops_multimodal_fusion.multilabel_margin_loss(Tensor input, Tensor target, int reduction=1) -> Tensor` | - | - | ✓ |
| multinomial | `torch.ops.ops_multimodal_fusion.multinomial(Tensor x, int num_samples, bool replacement=False, int seed=0) -> Tensor` | - | - | ✓ |
| nested_add_pad | `torch.ops.ops_multimodal_fusion.nested_add_pad(Tensor values, Tensor offsets, int max_L, Scalar padding_value=0) -> Tensor` | - | - | ✓ |
| nested_binary_op | `torch.ops.ops_multimodal_fusion.nested_binary_op(Tensor values, Tensor offsets, Tensor dense, int op_mode) -> Tensor` | - | - | ✓ |
| nested_bmm | `torch.ops.ops_multimodal_fusion.nested_bmm(Tensor a_values, Tensor b_values, Tensor sizes) -> Tensor` | - | - | ✓ |
| nested_remove_pad | `torch.ops.ops_multimodal_fusion.nested_remove_pad(Tensor padded, Tensor lengths) -> Tensor` | - | - | ✓ |
| nextafter | `torch.ops.ops_multimodal_fusion.nextafter(Tensor a, Tensor b) -> Tensor` | - | - | ✓ |
| norm | `torch.ops.ops_multimodal_fusion.norm(Tensor x, Scalar p, int dim, bool keepdim) -> Tensor` | - | - | ✓ |
| pdist | `torch.ops.ops_multimodal_fusion.pdist(Tensor input, float p=2.0) -> Tensor` | - | - | ✓ |
| pdist_backward | `torch.ops.ops_multimodal_fusion.pdist_backward(Tensor grad, Tensor input, float p, Tensor pdist_output) -> Tensor` | - | - | ✓ |
| poisson | `torch.ops.ops_multimodal_fusion.poisson(Tensor x, int seed=0) -> Tensor` | - | - | ✓ |
| polar | `torch.ops.ops_multimodal_fusion.polar(Tensor abs, Tensor angle) -> Tensor` | - | - | ✓ |
| polygamma | `torch.ops.ops_multimodal_fusion.polygamma(Tensor x, int n) -> Tensor` | - | - | ✓ |
| put | `torch.ops.ops_multimodal_fusion.put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor` | - | - | ✓ |
| quantized_relu | `torch.ops.ops_multimodal_fusion.quantized_relu(Tensor x, int zero_point) -> Tensor` | - | - | ✓ |
| rms_norm_gated | `torch.ops.ops_multimodal_fusion.rms_norm_gated(Tensor hidden_states, Tensor gate, Tensor gamma, float epsilon=1e-6) -> Tensor` | - | - | ✓ |
| rsqrt | `torch.ops.ops_multimodal_fusion.rsqrt(Tensor x) -> Tensor` | - | - | ✓ |
| scaled_modified_bessel_k0 | `torch.ops.ops_multimodal_fusion.scaled_modified_bessel_k0(Tensor x) -> Tensor` | - | - | ✓ |
| scaled_modified_bessel_k1 | `torch.ops.ops_multimodal_fusion.scaled_modified_bessel_k1(Tensor x) -> Tensor` | - | - | ✓ |
| searchsorted | `torch.ops.ops_multimodal_fusion.searchsorted(Tensor sorted_sequence, Tensor values, bool out_int32=False, bool right=False) -> Tensor` | - | - | ✓ |
| semi_structured_linear | `torch.ops.ops_multimodal_fusion.semi_structured_linear(Tensor x, Tensor w_compressed, Tensor metadata, Tensor bias, int m, int k) -> Tensor` | - | - | ✓ |
| shifted_chebyshev_polynomial_t | `torch.ops.ops_multimodal_fusion.shifted_chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| shifted_chebyshev_polynomial_u | `torch.ops.ops_multimodal_fusion.shifted_chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| shifted_chebyshev_polynomial_v | `torch.ops.ops_multimodal_fusion.shifted_chebyshev_polynomial_v(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| shifted_chebyshev_polynomial_w | `torch.ops.ops_multimodal_fusion.shifted_chebyshev_polynomial_w(Tensor x, Tensor n) -> Tensor` | - | - | ✓ |
| sigmoid | `torch.ops.ops_multimodal_fusion.sigmoid(Tensor x) -> Tensor` | - | - | ✓ |
| sin | `torch.ops.ops_multimodal_fusion.sin(Tensor x) -> Tensor` | - | - | ✓ |
| sinc | `torch.ops.ops_multimodal_fusion.sinc(Tensor x) -> Tensor` | - | - | ✓ |
| sparse_binary_intersect | `torch.ops.ops_multimodal_fusion.sparse_binary_intersect(Tensor indices_a, Tensor values_a, Tensor indices_b, Tensor values_b) -> Tensor[]` | - | - | ✓ |
| sparse_mask_intersection | `torch.ops.ops_multimodal_fusion.sparse_mask_intersection(Tensor source, Tensor mask_indices) -> Tensor` | - | - | ✓ |
| sparse_mask_projection | `torch.ops.ops_multimodal_fusion.sparse_mask_projection(Tensor sparse_indices, Tensor sparse_values, Tensor mask_indices) -> Tensor` | - | - | ✓ |
| spherical_bessel_j0 | `torch.ops.ops_multimodal_fusion.spherical_bessel_j0(Tensor x) -> Tensor` | - | - | ✓ |
| sqrt | `torch.ops.ops_multimodal_fusion.sqrt(Tensor x) -> Tensor` | - | - | ✓ |
| subm_conv | `torch.ops.ops_multimodal_fusion.subm_conv(Tensor features, Tensor neighbor, Tensor weight, Tensor? bias) -> Tensor` | - | - | ✓ |
| sum | `torch.ops.ops_multimodal_fusion.sum(Tensor x, int dim, bool keepdim) -> Tensor` | - | - | ✓ |
| swi_glu | `torch.ops.ops_multimodal_fusion.swi_glu(Tensor x) -> Tensor` | - | - | ✓ |
| take | `torch.ops.ops_multimodal_fusion.take(Tensor self, Tensor index) -> Tensor` | - | - | ✓ |
| tril_indices | `torch.ops.ops_multimodal_fusion.tril_indices(int row, int col, int offset, bool out_int32) -> Tensor` | - | - | ✓ |
| triu_indices | `torch.ops.ops_multimodal_fusion.triu_indices(int row, int col, int offset, bool out_int32) -> Tensor` | - | - | ✓ |
| unpack_pivots | `torch.ops.ops_multimodal_fusion.unpack_pivots(Tensor pivots, int perm_size) -> Tensor` | - | - | ✓ |
| upsample_linear1d | `torch.ops.ops_multimodal_fusion.upsample_linear1d(Tensor input, int output_size, bool align_corners=False, float scale=-1.) -> Tensor` | - | - | ✓ |
| upsample_nearest1d | `torch.ops.ops_multimodal_fusion.upsample_nearest1d(Tensor input, int output_size, float scale=-1.) -> Tensor` | - | - | ✓ |
| upsample_trilinear3d | `torch.ops.ops_multimodal_fusion.upsample_trilinear3d(Tensor input, int[3] output_size, bool align_corners=False, float scales_d=-1., float scales_h=-1., float scales_w=-1.) -> Tensor` | - | - | ✓ |
| weight_norm | `torch.ops.ops_multimodal_fusion.weight_norm(Tensor v, Tensor g, int dim) -> Tensor` | - | - | ✓ |
| zeta | `torch.ops.ops_multimodal_fusion.zeta(Tensor x, Tensor q) -> Tensor` | - | - | ✓ |
| nms_bev_iou | `torch.ops.ops_multimodal_fusion.nms_bev_iou(Tensor boxes_bev) -> Tensor` | - | - | ✓ |
