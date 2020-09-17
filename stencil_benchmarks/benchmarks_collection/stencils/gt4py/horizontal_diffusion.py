import numpy as np
import contextlib

from ..base import HorizontalDiffusionStencil
from ....tools import timing

from gt4py.testing.utils import build_dace_adhoc
from stencil_benchmarks.benchmark import Parameter
from gt4py.testing.utils import ApplyOTFOptimizer

#
# class GT4PyCPUStencilMixin:
#     def setup(self):
#         super().setup()
#         halo = (self.halo,) * 3
#         alignment = max(self.parameters["alignment"], 1)
#         passes = []
#         if self.parameters["use_otf_transform"]:
#             passes.append(ApplyOTFOptimizer())
#
#         self._gt4py_stencil_object = build_dace_adhoc(
#             definition=self.definition,
#             domain=self.domain,
#             halo=halo,
#             specialize_strides=self.strides,
#             dtype=self.parameters["dtype"],
#             passes=passes,
#             alignment=alignment,
#             layout=self.parameters["layout"],
#             loop_order=self.parameters["loop_order"],
#             device="cpu",
#         )


class GT4PyStencilMixin:
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
            device=self.device,
        )


import itertools
from gt4py import gtscript


class GT4PyDaceHorizontalDiffusionStencil(HorizontalDiffusionStencil):
    @timing.return_time
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
        return (
            exec_info["pyext_program_end_time"] - exec_info["pyext_program_start_time"]
        )

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


class CPUGT4PyDaceHorizontalDiffusionStencil(
    GT4PyStencilMixin, GT4PyDaceHorizontalDiffusionStencil
):
    device = "cpu"
    use_otf_transform = Parameter("use_otf_transform", True)
    loop_order = Parameter(
        "loop_order",
        default="IJK",
        choices=list("".join(p) for p in itertools.permutations("IJK")),
    )

    @contextlib.contextmanager
    def on_device(self, data):
        yield data


class GPUGT4PyDaceHorizontalDiffusionStencil(
    GT4PyStencilMixin, GT4PyDaceHorizontalDiffusionStencil
):
    device = "gpu"
    use_otf_transform = Parameter("use_otf_transform", True)
    loop_order = Parameter(
        "loop_order",
        default="IJK",
        choices=list("".join(p) for p in itertools.permutations("IJK")),
    )

    @property
    def backend(self):
        return "cuda"

    @contextlib.contextmanager
    def on_device(self, data):
        from ..cuda_hip import api
        from stencil_benchmarks.tools import array

        runtime = api.runtime(self.backend)

        device_data = [
            array.alloc_array(
                self.domain_with_halo,
                self.dtype,
                self.layout,
                self.alignment,
                index_to_align=(self.halo,) * 3,
                alloc=runtime.malloc,
                apply_offset=self.offset_allocations,
            )
            for _ in data
        ]

        for host_array, device_array in zip(data, device_data):
            runtime.memcpy(
                device_array.ctypes.data,
                host_array.ctypes.data,
                array.nbytes(host_array),
                "HostToDevice",
            )
        runtime.device_synchronize()

        for o in device_data:
            o.__cuda_array_interface__ = o.__array_interface__
        yield device_data

        for host_array, device_array in zip(data, device_data):
            runtime.memcpy(
                host_array.ctypes.data,
                device_array.ctypes.data,
                array.nbytes(host_array),
                "DeviceToHost",
            )
        runtime.device_synchronize()
