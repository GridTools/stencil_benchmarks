from stencil_benchmarks.benchmark import Parameter
from stencil_benchmarks.benchmarks_collection.stencils import base

from .mixin import StencilMixin
from .basic import BasicStencilMixin


class HorizontalDiffusionMixin(StencilMixin):
    block_size = Parameter('block size', (32, 8, 1))

    def template_file(self):
        return 'horizontal_diffusion_' + type(self).__name__.lower() + '.j2'

    def template_args(self):
        return dict(**super().template_args(), block_size=self.block_size)


class OnTheFly(BasicStencilMixin, base.HorizontalDiffusionStencil):
    def stencil_body(self):
        stride_x, stride_y, _ = self.strides
        return f'''const auto inp_ij = inp[index];
                const auto inp_im1j = inp[index - {stride_x}];
                const auto inp_ip1j = inp[index + {stride_x}];
                const auto inp_ijm1 = inp[index - {stride_y}];
                const auto inp_ijp1 = inp[index + {stride_y}];
                const auto inp_im2j = inp[index - 2 * {stride_x}];
                const auto inp_im1jm1 = inp[index - {stride_x} - {stride_y}];
                const auto inp_im1jp1 = inp[index - {stride_x} + {stride_y}];
                const auto inp_ip2j = inp[index + 2 * {stride_x}];
                const auto inp_ip1jm1 = inp[index + {stride_x} - {stride_y}];
                const auto inp_ip1jp1 = inp[index + {stride_x} + {stride_y}];
                const auto inp_ijm2 = inp[index - 2 * {stride_y}];
                const auto inp_ijp2 = inp[index + 2 * {stride_y}];

                const auto lap_ij = 4 * inp_ij - inp_im1j - inp_ip1j -
                inp_ijm1 - inp_ijp1;
                const auto lap_imj = 4 * inp_im1j - inp_im2j -
                    inp_ij - inp_im1jm1 -
                    inp_im1jp1;
                const auto lap_ipj = 4 * inp_ip1j - inp_ij -
                    inp_ip2j - inp_ip1jm1 -
                    inp_ip1jp1;
                const auto lap_ijm = 4 * inp_ijm1 - inp_im1jm1 -
                    inp_ip1jm1 - inp_ijm2 -
                    inp_ij;
                const auto lap_ijp = 4 * inp_ijp1 - inp_im1jp1 -
                    inp_ip1jp1 - inp_ij -
                    inp_ijp2;

                auto flx_ij = lap_ipj - lap_ij;
                flx_ij = flx_ij * (inp_ip1j - inp_ij) > 0 ? 0 : flx_ij;

                auto flx_imj = lap_ij - lap_imj;
                flx_imj = flx_imj * (inp_ij - inp_im1j) > 0 ? 0 : flx_imj;

                auto fly_ij = lap_ijp - lap_ij;
                fly_ij = fly_ij * (inp_ijp1 - inp_ij) > 0 ? 0 : fly_ij;

                auto fly_ijm = lap_ij - lap_ijm;
                fly_ijm = fly_ijm * (inp_ij - inp_ijm1) > 0 ? 0 : fly_ijm;

                out[index] = inp_ij - coeff[index] * (flx_ij - flx_imj +
                                        fly_ij - fly_ijm);
                '''


class OnTheFlyIncache(HorizontalDiffusionMixin,
                      base.HorizontalDiffusionStencil):
    pass


class Classic(HorizontalDiffusionMixin, base.HorizontalDiffusionStencil):
    pass


class JScanSharedMem(HorizontalDiffusionMixin,
                     base.HorizontalDiffusionStencil):
    pass


class JScanOtfIncache(HorizontalDiffusionMixin,
                      base.HorizontalDiffusionStencil):
    pass


class JScanOtf(HorizontalDiffusionMixin, base.HorizontalDiffusionStencil):
    pass


class JScanShuffleIncache(HorizontalDiffusionMixin,
                          base.HorizontalDiffusionStencil):
    pass


class JScanShuffle(HorizontalDiffusionMixin, base.HorizontalDiffusionStencil):
    pass


class JScanShuffleSystolic(HorizontalDiffusionMixin,
                           base.HorizontalDiffusionStencil):
    pass
