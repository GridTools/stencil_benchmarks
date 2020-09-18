from ..base import HorizontalDiffusionStencil
from ....tools import timing

from .mixin import CPUStencilMixin, GPUStencilMixin

from dace.codegen.instrumentation.report import InstrumentationReport
from gt4py import gtscript


class HorizontalDiffusionStencil(HorizontalDiffusionStencil):
    def run_stencil(self, data):
        with self.on_device(data) as device_data:
            exec_info = {}
            origin = (self.halo,) * 3
            inp, coeff, out = device_data
            self._gt4py_stencil_object.run(
                in_field=inp,
                coeff=coeff,
                out_field=out,
                exec_info=exec_info,
                _domain_=self.domain,
                _origin_=dict(in_field=origin, coeff=origin, out_field=origin),
            )
        report = InstrumentationReport(exec_info["instrumentation_report"])
        total_ms = sum(sum(v) for v in report.entries.values())
        return total_ms / 1000
        #         (
        #     total_ms
        #     / 1000  # exec_info["pyext_program_end_time"] - exec_info["pyext_program_start_time"]
        # )

    @property
    def definition(self):
        def horizontal_diffusion(
            in_field: gtscript.Field["dtype"],
            out_field: gtscript.Field["dtype"],
            coeff: gtscript.Field["dtype"],
        ):
            with computation(PARALLEL), interval(...):
                lap_field = 4.0 * in_field[0, 0, 0] - (
                    in_field[1, 0, 0]
                    + in_field[-1, 0, 0]
                    + in_field[0, 1, 0]
                    + in_field[0, -1, 0]
                )
                res1 = lap_field[1, 0, 0] - lap_field[0, 0, 0]
                flx_field = (
                    0 if (res1 * (in_field[1, 0, 0] - in_field[0, 0, 0])) > 0 else res1
                )
                res2 = lap_field[0, 1, 0] - lap_field[0, 0, 0]
                fly_field = (
                    0 if (res2 * (in_field[0, 1, 0] - in_field[0, 0, 0])) > 0 else res2
                )
                out_field = in_field[0, 0, 0] - coeff[0, 0, 0] * (
                    flx_field[0, 0, 0]
                    - flx_field[-1, 0, 0]
                    + fly_field[0, 0, 0]
                    - fly_field[0, -1, 0]
                )

        return horizontal_diffusion


class GpuHorizontalDiffusionStencil(GPUStencilMixin, HorizontalDiffusionStencil):
    pass


class CpuHorizontalDiffusionStencil(CPUStencilMixin, HorizontalDiffusionStencil):
    pass
