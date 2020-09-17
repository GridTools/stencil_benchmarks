import numpy as np

from ..base import HorizontalDiffusionStencil
from ....tools import timing

from gt4py.testing.utils import build_dace_adhoc
from stencil_benchmarks.benchmark import Parameter
from gt4py.testing.utils import ApplyOTFOptimizer


class GT4PyCPUStencilMixin:
    def setup(self):
        super().setup()
        halo = (self.halo,) * 3
        alignment = max(self.parameters["alignment"], 1)
        passes = []
        if self.parameters["use_otf_transform"]:
            passes.append(ApplyOTFOptimizer())

        self._gt4py_stencil_object = build_dace_adhoc(
            definition=self.definition,
            domain=self.domain,
            halo=halo,
            specialize_strides=self.strides,
            dtype=self.parameters["dtype"],
            passes=passes,
            alignment=alignment,
            layout=self.parameters["layout"],
            loop_order=self.parameters["loop_order"],
            device="cpu",
        )


class GT4PyGPUStencilMixin:
    def setup(self):
        super().setup()
        halo = (self.halo,) * 3
        passes = []

        if self.parameters["use_otf_transform"]:
            passes.append(ApplyOTFOptimizer())

        self._gt4py_stencil_object = build_dace_adhoc(
            definition=self.definition,
            domain=self.domain,
            halo=halo,
            specialize_strides=self.strides,
            dtype=self.parameters["dtype"],
            passes=passes,
            alignment=self.parameters["alignment"],
            layout=self.parameters["layout"],
            loop_order=self.parameters["loop_order"],
            device="gpu",
        )


import itertools
from gt4py import gtscript


class GT4PyDaceHorizontalDiffusionStencil(
    HorizontalDiffusionStencil
):

    @timing.return_time
    def run_stencil(self, data):
        import gt4py.storage

        backend = self._gt4py_stencil_object.backend
        origin = (self.halo,) * 3
        inp, coeff, out = [
            gt4py.storage.from_array(
                d, backend=backend, default_origin=origin, dtype=d.dtype
            )
            for d in data
        ]
        self._gt4py_stencil_object.run(
            in_field=inp,
            coeff=coeff,
            out_field=out,
            exec_info=None,
            _domain_=self.domain,
            _origin_=dict(in_field=origin, coeff=origin, out_field=origin),
        )
        inp.device_to_host(force=True)
        data[0][...] = inp
        coeff.device_to_host(force=True)
        data[1][...] = coeff
        out.device_to_host(force=True)
        data[2][...] = out

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
class CPUGT4PyDaceHorizontalDiffusionStencil(GT4PyCPUStencilMixin, GT4PyDaceHorizontalDiffusionStencil):
    use_otf_transform = Parameter("use_otf_transform", True)
    loop_order = Parameter(
        "loop_order",
        default="IJK",
        choices=list("".join(p) for p in itertools.permutations("IJK")),
    )
class GPUGT4PyDaceHorizontalDiffusionStencil(GT4PyGPUStencilMixin, GT4PyDaceHorizontalDiffusionStencil):
    use_otf_transform = Parameter("use_otf_transform", True)
    loop_order = Parameter(
        "loop_order",
        default="IJK",
        choices=list("".join(p) for p in itertools.permutations("IJK")),
    )