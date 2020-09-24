import contextlib

import dace
from dace.codegen.instrumentation.report import InstrumentationReport
from gt4py.testing.utils import (build_dace_adhoc, ApplyOTFOptimizer,
                                 DeduplicateAccesses, PrefetchingKCaches,
                                 PruneTransientOutputs, SubgraphFusion)
from stencil_benchmarks.benchmark import Parameter, Benchmark


class StencilMixin(Benchmark):
    use_prune_transient_outputs = Parameter(
        'switch pruning of transient output accesses on or off', default=True)
    use_deduplicate_accesses = Parameter(
        'only access each offset once per map and field.', default=False)
    use_otf_transform = Parameter('apply on-the-fly transform', default=True)
    use_subgraph_fusion = Parameter('fuse subgraphs', default=True)
    use_prefetching = Parameter('apply prefetching transform', default=False)
    prefetch_arrays = Parameter('arrays to prefetch', default='')
    device = Parameter('DaCe device to use', 'cpu', choices=['cpu', 'gpu'])
    backend = Parameter('Dace backend to use',
                        'default',
                        choices=['default', 'cuda', 'hip'])
    loop_order = Parameter(
        'loop order, 2 means innermost dimension, 0 outermost', (1, 0, 2))
    dry_runs = Parameter('kernel dry-runs before the measurement', 0)
    block_size = Parameter('GPU block size', (0, 0, 0))

    def setup(self):
        super().setup()
        halo = (self.halo, ) * 3
        alignment = max(self.alignment, 1)
        passes = []
        if self.use_prune_transient_outputs:
            passes.append(PruneTransientOutputs())
        if self.use_otf_transform:
            passes.append(ApplyOTFOptimizer())
        if self.use_subgraph_fusion:
            passes.append(
                SubgraphFusion(storage_type=dace.dtypes.StorageType.Register))
        if self.use_deduplicate_accesses:
            passes.append(DeduplicateAccesses())
        if self.use_prefetching:
            arrays = self.prefetch_arrays.split(',')
            passes.append(PrefetchingKCaches(arrays=arrays))

        kwargs = {}
        if self.backend != 'default':
            kwargs['backend'] = self.backend
        if hasattr(self, 'constants'):
            kwargs['constants'] = self.constants
        if self.block_size != (0, 0, 0):
            kwargs['gpu_block_size'] = ','.join(
                str(b) for b in self.block_size)
        self._gt4py_stencil_object = build_dace_adhoc(
            definition=self.definition,
            domain=self.domain,
            halo=halo,
            specialize_strides=self.strides,
            dtype=self.dtype,
            passes=passes,
            alignment=alignment,
            layout=self.layout,
            loop_order=''.join('IJK'[self.loop_order.index(i)]
                               for i in range(3)),
            device=self.device,
            **kwargs)

    @contextlib.contextmanager
    def on_device(self, data):
        if self.device == 'cpu':
            yield data
            return

        from ..cuda_hip import api
        from stencil_benchmarks.tools import array

        runtime = api.runtime(self.backend)

        device_data = [
            array.alloc_array(
                self.domain_with_halo,
                self.dtype,
                self.layout,
                self.alignment,
                index_to_align=(self.halo, ) * 3,
                alloc=runtime.malloc,
                apply_offset=self.offset_allocations,
            ) for _ in data
        ]

        for host_array, device_array in zip(data, device_data):
            runtime.memcpy(
                device_array.ctypes.data,
                host_array.ctypes.data,
                array.nbytes(host_array),
                'HostToDevice',
            )
        runtime.device_synchronize()
        from types import SimpleNamespace

        device_data_wrapped = [
            SimpleNamespace(__cuda_array_interface__=o.__array_interface__)
            for o in device_data
        ]

        yield device_data_wrapped

        for host_array, device_array in zip(data, device_data):
            runtime.memcpy(
                host_array.ctypes.data,
                device_array.ctypes.data,
                array.nbytes(host_array),
                'DeviceToHost',
            )
        runtime.device_synchronize()

    def gt4py_data(self, data):
        return dict(zip(self.args, data))

    def run_stencil(self, data):
        with self.on_device(data) as device_data:
            exec_info = {}
            origin = (self.halo, ) * 3
            for _ in range(self.dry_runs + 1):
                self._gt4py_stencil_object.run(**self.gt4py_data(device_data),
                                               exec_info=exec_info,
                                               _domain_=self.domain,
                                               _origin_=dict.fromkeys(
                                                   self.args, origin))
        report = InstrumentationReport(exec_info['instrumentation_report'])
        total_ms = sum(sum(v) for v in report.entries.values())
        return total_ms / 1000
