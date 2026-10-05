import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  defaultSidebar: [
    {
      type: 'doc',
      id: 'introduction',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Getting Started',
      items: ['examples', 'models', 'onnx'],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/ir-design',
        'architecture/pipeline',
        'architecture/jit-graphs',
        'architecture/kernel-origins',
        {
          type: 'category',
          label: 'Codegen Pipeline',
          items: [
            'architecture/codegen/overview',
            'architecture/codegen/rangeify',
            'architecture/codegen/expander',
            'architecture/codegen/devectorizer',
            'architecture/codegen/linearizer',
            'architecture/codegen/worked-example',
          ],
        },
        {
          type: 'category',
          label: 'Optimizations',
          items: [
            'architecture/optimizations/pattern-system',
            'architecture/optimizations/algebraic-simplification',
            'architecture/optimizations/index-arithmetic',
            'architecture/optimizations/range-optimization',
            'architecture/optimizations/strength-reduction',
            'architecture/optimizations/kernel-search',
          ],
        },
        'architecture/op-bestiary',
      ],
    },
    {
      type: 'category',
      label: 'Tile Kernels (tk)',
      items: [
        'tile-kernels/overview',
        'tile-kernels/where-flops-hide',
        'tile-kernels/tiling',
        'tile-kernels/lowering',
        'tile-kernels/first-kernel',
        'tile-kernels/builder-reference',
        'tile-kernels/wave-portability',
        'tile-kernels/kernel-library',
        'tile-kernels/flash-attention',
        'tile-kernels/tuning',
        'tile-kernels/debugging',
        'tile-kernels/profiling',
        'tile-kernels/comparison',
      ],
    },
    {
      type: 'category',
      label: 'Backends',
      items: [
        'backends/overview',
        'backends/cpu',
        'backends/jit-loader',
        'backends/metal',
        {
          type: 'category',
          label: 'AMD Backend',
          items: [
            'backends/amd/overview',
            'backends/amd/kfd-bindings',
            'backends/amd/queues-and-dispatch',
            'backends/amd/compile-and-graph',
            'backends/amd/am-driver',
            'backends/amd/debugging',
          ],
        },
        {
          type: 'category',
          label: 'CUDA Backend',
          items: [
            'backends/cuda/overview',
            'backends/cuda/architecture',
            'backends/cuda/codegen',
            'backends/cuda/profiling',
            'backends/cuda/limitations',
            'backends/cuda/debugging',
          ],
        },
      ],
    },
  ],
};

export default sidebars;
