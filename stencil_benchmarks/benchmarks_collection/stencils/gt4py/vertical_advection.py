from ..base import VerticalAdvectionStencil
from ....tools import timing

from gt4py import gtscript
from .mixin import CPUStencilMixin, GPUStencilMixin


class VerticalAdvectionStencil(VerticalAdvectionStencil):
    @timing.return_time
    def run_stencil(self, data):
        with self.on_device(data) as device_data:
            exec_info = {}
            origin = (self.halo,) * 3
            ccol, _, _, upos, ustage, utens, utens_stage, wcon = device_data
            print(data.datacol.shape)
            dtr_stage = 3.0 / 20.0
            self._gt4py_stencil_object.run(
                utens_stage=utens_stage,
                u_stage=ustage,
                wcon=wcon,
                u_pos=upos,
                utens=utens,
                dtr_stage=dtr_stage,
                exec_info=exec_info,
                _domain_=self.domain,
                _origin_=dict(
                    utens_stage=origin,
                    u_stage=origin,
                    wcon=origin,
                    u_pos=origin,
                    utens=origin,
                ),
            )
        return (
            exec_info["pyext_program_end_time"] - exec_info["pyext_program_start_time"]
        )

    @property
    def constants(self):
        return {"BET_M": 0.5, "BET_P": 0.5}

    @property
    def definition(self):
        def vertical_advection_dycore(
            utens_stage: gtscript.Field["dtype"],
            u_stage: gtscript.Field["dtype"],
            wcon: gtscript.Field["dtype"],
            u_pos: gtscript.Field["dtype"],
            utens: gtscript.Field["dtype"],
            *,
            dtr_stage: float,
        ):
            from __externals__ import BET_M, BET_P

            with computation(FORWARD):
                with interval(0, 1):
                    gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])
                    cs = gcv * BET_M

                    ccol = gcv * BET_P
                    bcol = dtr_stage - ccol[0, 0, 0]

                    # update the d column
                    correction_term = -cs * (u_stage[0, 0, 1] - u_stage[0, 0, 0])
                    dcol = (
                        dtr_stage * u_pos[0, 0, 0]
                        + utens[0, 0, 0]
                        + utens_stage[0, 0, 0]
                        + correction_term
                    )

                    # Thomas forward
                    divided = 1.0 / bcol[0, 0, 0]
                    ccol = ccol[0, 0, 0] * divided
                    dcol = dcol[0, 0, 0] * divided

                with interval(1, -1):
                    gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
                    gcv = 0.25 * (wcon[1, 0, 1] + wcon[0, 0, 1])

                    as_ = gav * BET_M
                    cs = gcv * BET_M

                    acol = gav * BET_P
                    ccol = gcv * BET_P
                    bcol = dtr_stage - acol[0, 0, 0] - ccol[0, 0, 0]

                    # update the d column
                    correction_term = -as_ * (
                        u_stage[0, 0, -1] - u_stage[0, 0, 0]
                    ) - cs * (u_stage[0, 0, 1] - u_stage[0, 0, 0])
                    dcol = (
                        dtr_stage * u_pos[0, 0, 0]
                        + utens[0, 0, 0]
                        + utens_stage[0, 0, 0]
                        + correction_term
                    )

                    # Thomas forward
                    divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
                    ccol = ccol[0, 0, 0] * divided
                    dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

                with interval(-1, None):
                    gav = -0.25 * (wcon[1, 0, 0] + wcon[0, 0, 0])
                    as_ = gav * BET_M
                    acol = gav * BET_P
                    bcol = dtr_stage - acol[0, 0, 0]

                    # update the d column
                    correction_term = -as_ * (u_stage[0, 0, -1] - u_stage[0, 0, 0])
                    dcol = (
                        dtr_stage * u_pos[0, 0, 0]
                        + utens[0, 0, 0]
                        + utens_stage[0, 0, 0]
                        + correction_term
                    )

                    # Thomas forward
                    divided = 1.0 / (bcol[0, 0, 0] - ccol[0, 0, -1] * acol[0, 0, 0])
                    dcol = (dcol[0, 0, 0] - (dcol[0, 0, -1]) * acol[0, 0, 0]) * divided

            with computation(BACKWARD):
                with interval(-1, None):
                    datacol = dcol[0, 0, 0]
                    data_col = datacol
                    utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])

                with interval(0, -1):
                    datacol = dcol[0, 0, 0] - ccol[0, 0, 0] * data_col[0, 0, 1]
                    data_col = datacol
                    utens_stage = dtr_stage * (datacol - u_pos[0, 0, 0])

        return vertical_advection_dycore


class GpuVerticalAdvectionStencil(GPUStencilMixin, VerticalAdvectionStencil):
    def setup(self):
        assert self.parameters["u_only"]
        return super().setup()


class CpuVerticalAdvectionStencil(CPUStencilMixin, VerticalAdvectionStencil):
    def setup(self):
        assert self.parameters["u_only"]
        return super().setup()
