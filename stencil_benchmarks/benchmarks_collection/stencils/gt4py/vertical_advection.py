from stencil_benchmarks.benchmark import ParameterError
from stencil_benchmarks.benchmarks_collection.stencils import base

from gt4py import gtscript
from .mixin import StencilMixin


class Stencil(StencilMixin, base.VerticalAdvectionStencil):
    def setup(self):
        super().setup()
        if not self.u_only:
            raise ParameterError('only --u-only is currently supported')

    def gt4py_data(self, data):
        ustage, upos, utens, utensstage, wcon, ccol, dcol, datacol = data
        return dict(ustage=ustage,
                    upos=upos,
                    utens=utens,
                    utensstage=utensstage,
                    wcon=wcon)

    @property
    def constants(self):
        return {'BET_M': 0.5, 'BET_P': 0.5, 'dtr_stage': 3 / 20}

    @property
    def definition(self):
        def vertical_advection_dycore(
                utensstage: gtscript.Field['dtype'],  # noqa: 821
                ustage: gtscript.Field['dtype'],  # noqa: 821
                wcon: gtscript.Field['dtype'],  # noqa: 821
                upos: gtscript.Field['dtype'],  # noqa: 821
                utens: gtscript.Field['dtype'],  # noqa: 821
        ):
            from __externals__ import BET_M, BET_P, dtr_stage

            with computation(FORWARD):  # noqa: 821
                with interval(0, 1):  # noqa: 821
                    gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])
                    cs = gcv * BET_M

                    ccol = gcv * BET_P
                    bcol = dtr_stage - ccol[0, 0, 0]

                    # update the d column
                    correction_term = -cs * (ustage[0, 0, 1] - ustage[0, 0, 0])
                    dcol = (dtr_stage * upos[0, 0, 0] + utens[0, 0, 0] +
                            utensstage[0, 0, 0] + correction_term)

                    # Thomas forward
                    divided = 1.0 / bcol[0, 0, 0]
                    ccol = ccol[0, 0, 0] * divided
                    dcol = dcol[0, 0, 0] * divided

                with interval(1, -1):  # noqa: 821
                    gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
                    gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])

                    as_ = gav * BET_M
                    cs = gcv * BET_M

                    acol = gav * BET_P
                    ccol = gcv * BET_P
                    bcol = dtr_stage - acol[0, 0, 0] - ccol[0, 0, 0]

                    # update the d column
                    correction_term = -as_ * (ustage[0, 0, -1] - ustage[
                        0, 0, 0]) - cs * (ustage[0, 0, 1] - ustage[0, 0, 0])
                    dcol = (dtr_stage * upos[0, 0, 0] + utens[0, 0, 0] +
                            utensstage[0, 0, 0] + correction_term)

                    # Thomas forward
                    divided = 1.0 / (bcol[0, 0, 0] -
                                     ccol[0, 0, -1] * acol[0, 0, 0])
                    ccol = ccol[0, 0, 0] * divided
                    dcol = (dcol[0, 0, 0] -
                            (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

                with interval(-1, None):  # noqa: 821
                    gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
                    as_ = gav * BET_M
                    acol = gav * BET_P
                    bcol = dtr_stage - acol[0, 0, 0]

                    # update the d column
                    correction_term = -as_ * (ustage[0, 0, -1] -
                                              ustage[0, 0, 0])
                    dcol = (dtr_stage * upos[0, 0, 0] + utens[0, 0, 0] +
                            utensstage[0, 0, 0] + correction_term)

                    # Thomas forward
                    divided = 1.0 / (bcol[0, 0, 0] -
                                     ccol[0, 0, -1] * acol[0, 0, 0])
                    dcol = (dcol[0, 0, 0] -
                            (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

            with computation(BACKWARD):  # noqa: 821
                with interval(-1, None):  # noqa: 821
                    datacol = dcol[0, 0, 0]
                    data_col = datacol
                    utensstage = dtr_stage * (datacol - upos[0, 0, 0])

                with interval(0, -1):  # noqa: 821
                    datacol = dcol[0, 0, 0] - ccol[0, 0, 0] * data_col[0, 0, 1]
                    data_col = datacol
                    utensstage = dtr_stage * (datacol - upos[0, 0, 0])

        return vertical_advection_dycore
