import {
  BarChart,
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasAction,
  useHostTheme,
  useState,
} from "cursor/canvas";

type Category = "Pointwise" | "Reduction" | "Persistent";

const corpus = [
  { category: "Pointwise", files: 624, exact: 397, structural: 195, selected: 27 },
  { category: "Reduction", files: 284, exact: 210, structural: 165, selected: 20 },
  { category: "Persistent", files: 129, exact: 97, structural: 55, selected: 25 },
];

const familyGroups: Record<Category, Array<[string, number, string]>> = {
  Pointwise: [
    ["Linear casts", 2, "Downcast/upcast byte ratios; one tail-masked body"],
    ["Layout materialization", 2, "Slice and transpose div/mod address decomposition"],
    ["GELU, SiLU, ReLU", 3, "Different transcendental and in-place compute chains"],
    ["Activation backward + MM", 1, "Gathered SiLU-gradient epilogue"],
    ["Softmax stages", 2, "Online preparation versus split/unbind epilogue"],
    ["Indirect scatter", 2, "Loaded-index store versus conditional mutated scatter"],
    ["Embedding gather", 1, "Indirect table load with bounds assertion"],
    ["RoPE", 2, "Frequency-table generation versus cos/sin application"],
    ["Mask generation", 1, "Bitwise causal indexing and where"],
    ["Sigmoid/gating fan-out", 2, "Residual sigmoid and five-output clamped gate"],
    ["MM/stack epilogue", 1, "Gather and stack of matrix-multiply outputs"],
    ["Pure fan-out", 1, "One load feeding three stores"],
    ["Norm epilogues", 2, "RMSNorm mega-fusion and LayerNorm backward"],
    ["Extreme fan-in", 1, "Dozens of pointers; register-pressure stress"],
    ["Attention backward", 1, "Large permute/slice materialization"],
    ["Scalar boundary", 1, "Degenerate xnumel=1 launch"],
    ["Complex layout", 1, "view_as_complex reinterpretation"],
    ["rsqrt micro-fusion", 1, "Mean/pow/rsqrt with indexing"],
  ],
  Reduction: [
    ["LLM RMS/variance", 3, "Two RMS widths plus a distinct variance-sum epilogue"],
    ["Mul-sum regimes", 3, "OUTER, large-x OUTER, and large-r DEFAULT"],
    ["Welford LayerNorm", 3, "Compact, cat-fused, and convolution/layout-fused"],
    ["Norm backward", 2, "LayerNorm epilogue and unique four-loop RMS backward"],
    ["Diffusion RMSNorm", 1, "Large-r permuted normalization"],
    ["Bounded slice sum", 1, "Conditional OUTER load with slice-backward indexing"],
    ["Activation gradients", 2, "GELU and sole SiLU reduction"],
    ["Online softmax", 1, "Streaming max/sum accumulator"],
    ["Atomic embedding", 1, "Sum followed by indirect atomic_add"],
    ["Extreme fan-in", 1, "79-input cat-to-sum body"],
    ["Three-loop variance", 1, "Deep fused pre/post reduction chain"],
    ["OUTER_TINY", 1, "Only non-persistent micro-reduction in the corpus"],
  ],
  Persistent: [
    ["Softmax", 4, "Full, causal, small-r masked, and prep-only forms"],
    ["RMSNorm / LayerNorm forward", 5, "In-place, large, transposed, split, and affine IO"],
    ["Normalization backward", 4, "Core, slice, r=2, and padded-r=14 boundary bodies"],
    ["Indexed reductions", 2, "Gather/assert and post-sum indexed load"],
    ["Expanded statistic", 1, "Large-x div/pow/sum fusion"],
    ["Layout / slice sums", 2, "Padded OUTER_TINY and slice-backward addressing"],
    ["Outer and inner sums", 2, "OUTER r=4 and degenerate INNER x=1"],
    ["In-place moment", 1, "Five-IO variance accumulation"],
    ["Masked ReLU sum", 1, "Non-softmax causal mask and reduction"],
    ["Mean-of-powers", 1, "Moment reduction without rsqrt"],
    ["Tiny scalar reduction", 1, "OUTER_TINY x=4, r=4 scheduling boundary"],
    ["SiLU/select backward", 1, "Dual reduction with seven-tensor IO"],
  ],
};

function SummaryCard({
  title,
  count,
  children,
}: {
  title: string;
  count: string;
  children: string;
}) {
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm">{count}</Pill>}>{title}</CardHeader>
      <CardBody>
        <Text>{children}</Text>
      </CardBody>
    </Card>
  );
}

export default function PriorityKernelDiversity() {
  const [category, setCategory] = useState<Category>("Pointwise");
  const dispatch = useCanvasAction();
  const theme = useHostTheme();
  const rows = familyGroups[category].map(([family, slots, distinction]) => [
    family,
    slots,
    distinction,
  ]);

  return (
    <Stack gap={20} style={{ padding: 24, background: theme.bg.editor }}>
      <Stack gap={6}>
        <H1>Priority kernel diversity</H1>
        <Text tone="secondary">
          Exhaustive body-and-input analysis of 1,037 Inductor kernels across 15
          model captures. The selected unit is an optimization mechanism, not a
          filename or model occurrence.
        </Text>
      </Stack>

      <Row gap={28} wrap>
        <Stat value="1,037" label="source kernels parsed" />
        <Stat value="72" label="representatives selected" tone="info" />
        <Stat value="0" label="exact-body duplicates selected" tone="success" />
        <Stat value="14.4×" label="corpus compression" />
      </Row>

      <Callout tone="info" title="Selection rule">
        A new slot requires a changed algorithm, address topology, reduction
        hint, loop depth, IO contract, mask, or block-padding regime. Model name,
        graph ordinal, hash suffix, and an ordinary xnumel change do not qualify.
      </Callout>

      <Grid columns="1.35fr 1fr" gap={16}>
        <Card>
          <CardHeader>Raw files versus deduplicated structure</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <H3>Kernel count by category and similarity level</H3>
              <BarChart
                categories={corpus.map((item) => item.category)}
                series={[
                  { name: "Source files", data: corpus.map((item) => item.files) },
                  { name: "Exact bodies", data: corpus.map((item) => item.exact) },
                  {
                    name: "Shape-normalized structures",
                    data: corpus.map((item) => item.structural),
                  },
                  { name: "Selected", data: corpus.map((item) => item.selected) },
                ]}
                height={260}
                showValues
              />
              <Text size="small" tone="tertiary">
                X-axis: kernel category. Y-axis: kernel count. Exact bodies keep
                shape constants; structural counts normalize shapes and SSA
                names. Source: priority capture, analyzed 2026-09-17.
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader>Reduction scheduling regimes</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <H3>Files by ReductionHint</H3>
              <BarChart
                categories={["Reduction", "Persistent"]}
                series={[
                  { name: "INNER", data: [112, 76] },
                  { name: "DEFAULT", data: [89, 12] },
                  { name: "OUTER", data: [82, 29] },
                  { name: "OUTER_TINY", data: [1, 12] },
                ]}
                height={260}
                stacked
              />
              <Text size="small" tone="tertiary">
                X-axis: reduction category. Y-axis: source kernel count. Stacks
                show exact decorator hints. Source: all 413 reduction kernels,
                analyzed 2026-09-17.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={10}>
        <H2>What dominates each category</H2>
        <Grid columns={3} gap={12}>
          <SummaryCard title="Pointwise" count="624 → 27">
            279 simple casts dominate, but useful diversity comes from layout
            decomposition, gather/scatter, masks, RoPE, nonlinear math,
            fan-in/fan-out, mutation, and backward epilogues. All are Grid1D;
            no RNG body appears.
          </SummaryCard>
          <SummaryCard title="Reduction" count="284 → 20">
            Every body has one r0 axis. Sum-based reductions account for 224
            files; Welford, RMSNorm, online softmax, atomic embedding, loop
            depth, and INNER/OUTER geometry create the distinct families.
          </SummaryCard>
          <SummaryCard title="Persistent" count="129 → 25">
            The meaningful boundaries are tiny outer reductions, padded RBLOCK,
            online softmax, normalization IO contracts, and indexed or masked
            reductions. Thirty bodies pad RBLOCK beyond logical r0.
          </SummaryCard>
        </Grid>
      </Stack>

      <Stack gap={10}>
        <Row gap={8} align="center" wrap>
          <H2>Representative mechanisms</H2>
          <Pill
            active={category === "Pointwise"}
            onClick={() => setCategory("Pointwise")}
          >
            Pointwise
          </Pill>
          <Pill
            active={category === "Reduction"}
            onClick={() => setCategory("Reduction")}
          >
            Reduction
          </Pill>
          <Pill
            active={category === "Persistent"}
            onClick={() => setCategory("Persistent")}
          >
            Persistent
          </Pill>
        </Row>
        <Table
          headers={["Mechanism group", "Slots", "Why these are distinct"]}
          rows={rows}
          columnAlign={["left", "right", "left"]}
          striped
          stickyHeader
        />
        <Text size="small" tone="tertiary">
          Slot groups are mutually exclusive within each selected category. The
          exact 72 relative paths are stored with this artifact.
        </Text>
      </Stack>

      <Grid columns={2} gap={16}>
        <Stack gap={8}>
          <H2>Why model-balanced sampling fails</H2>
          <Text>
            GPT-OSS 20B and 120B have 100% shape-normalized overlap in every
            category. DeepSeek Flash/Pro overlap by roughly 96% pointwise, 69%
            reduction, and 79% persistent. Wan 2.1/2.2 overlap by roughly 81%
            pointwise and 78% reduction.
          </Text>
          <Text tone="secondary">
            Giving each model equal quota therefore repeats whole optimization
            families. Model identity is useful for regression validation, not
            for defining the optimization queue.
          </Text>
        </Stack>
        <Stack gap={8}>
          <H2>Coverage gaps in this corpus</H2>
          <Text>
            No pointwise RNG, true r0×r1 reduction, reduction-only max,
            persistent Welford, persistent atomic, persistent tl.dot, or
            log-softmax body is present. Add these from another corpus rather
            than manufacturing more slots from existing shape variants.
          </Text>
        </Stack>
      </Grid>

      <Row gap={8} wrap>
        <Button
          variant="primary"
          onClick={() =>
            dispatch({
              type: "openFile",
              path: "/home/niromero/docker_workspace/framework_scripts/inductor/kernels/priority/priority_representative_kernel_paths.txt",
            })
          }
        >
          Open exact path list
        </Button>
        <Button
          variant="secondary"
          onClick={() =>
            dispatch({
              type: "openFile",
              path: "/home/niromero/docker_workspace/framework_scripts/inductor/kernels/priority/priority-kernel-diversity.md",
            })
          }
        >
          Open methodology and PAAS commands
        </Button>
      </Row>
    </Stack>
  );
}
