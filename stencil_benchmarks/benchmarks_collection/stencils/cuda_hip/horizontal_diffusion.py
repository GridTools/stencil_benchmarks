from .mixin import BasicStencilMixin, HorizontalDiffusionMixin
from ..base import HorizontalDiffusionStencil
from .... import benchmark


class OnTheFly(BasicStencilMixin, HorizontalDiffusionStencil):
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


class Classic(HorizontalDiffusionMixin, HorizontalDiffusionStencil):
    def setup(self):
        super().setup()

        if self.block_size[0] < self.block_size[1] + 2:
            raise benchmark.ParameterError('unsupported block size')


class JScan(HorizontalDiffusionMixin, HorizontalDiffusionStencil):
    pass


class JScanOtf(HorizontalDiffusionMixin, HorizontalDiffusionStencil):
    pass


class JScanShuffle(HorizontalDiffusionMixin, HorizontalDiffusionStencil):
    pass
