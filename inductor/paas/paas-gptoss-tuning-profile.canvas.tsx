import {
  BarChart,
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Row,
  Stack,
  Stat,
  Text,
  useCanvasAction,
  useHostTheme,
} from "cursor/canvas";

const timeCategories = [
  "Process startup",
  "Validation",
  "Timeouts",
  "Shutdown + detection",
  "Confirmation",
  "Input allocation",
  "Triton timing",
  "Everything else",
];

const timeMinutes = [
  22.801,
  20.633,
  19.639,
  7.459,
  0.859,
  0.514,
  0.232,
  0.312,
];

const xblockCategories = [
  "1",
  "2",
  "4",
  "8",
  "16",
  "32",
  "64",
  "128",
  "256",
  "512",
  "1024",
  "2048",
  "4096",
  "8192",
];

const xblockMinutes = [
  2.636,
  2.569,
  2.577,
  2.594,
  2.602,
  2.661,
  2.835,
  3.66,
  6.003,
  3.952,
  5.228,
  8.337,
  12.238,
  13.396,
];

const xblockTimeouts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 9];

const validationCategories = [
  "Candidate compile/load + launch",
  "Reference compile/load + launch",
  "FP64 compare + restore",
  "Snapshots + copies",
];

const validationMinutes = [12.883, 6.36, 1.381, 0.004];

const savingsCategories = [
  "8-GPU config sharding",
  "16K reduction cap",
  "Polling upper bound",
  "All Triton timing",
];

const savingsMinutes = [62.336, 38.514, 4.008, 0.232];

export default function PaasGptossTuningProfile() {
  const theme = useHostTheme();
  const dispatch = useCanvasAction();
  const reportPath =
    "/home/niromero/docker_workspace/framework_scripts/inductor/paas/PAAS_GFX942_GPTOSS_TUNING_TIME_PROFILE.md";

  return (
    <Stack
      gap={24}
      style={{
        maxWidth: 1180,
        margin: "0 auto",
        padding: 24,
        background: theme.bg.editor,
      }}
    >
      <Row align="start" justify="space-between" gap={20} wrap>
        <Stack gap={6} style={{ minWidth: 280 }}>
          <H1>PAAS tuning time profile</H1>
          <Text tone="secondary">
            One clean-cache gpt-oss reduction run on MI308X / gfx942
          </Text>
          <Text size="small" tone="tertiary">
            Source: 7,064 profile events · trunk ad6d4416 · September 17,
            2026
          </Text>
        </Stack>
        <Button
          variant="secondary"
          onClick={() => dispatch({ type: "openFile", path: reportPath })}
        >
          Open full report
        </Button>
      </Row>

      <Callout tone="info" title="The benchmark loop is not the bottleneck">
        PAAS spent only 13.9 seconds in Triton timing. Fresh process startup,
        validation, timeout waits, and process shutdown consumed 97.4% of the
        1 hour 12 minute run.
      </Callout>

      <Grid
        columns="repeat(auto-fit, minmax(180px, 1fr))"
        gap={16}
        style={{ alignItems: "stretch" }}
      >
        <Stat value="1h 12m 26.9s" label="Total tuning time" />
        <Stat value="500" label="Configurations tested" />
        <Stat value="19" label="Timeouts" tone="danger" />
        <Stat value="0.32%" label="Time in Triton timing" tone="info" />
      </Grid>

      <Divider />

      <H2>Where the time went</H2>
      <Grid columns="minmax(0, 1.45fr) minmax(280px, 0.75fr)" gap={18}>
        <Card size="lg">
          <CardHeader trailing="Minutes">Exclusive wall-time buckets</CardHeader>
          <CardBody>
            <Text size="small" tone="tertiary">
              Category axis: tuning activity · Value axis: wall time (minutes)
            </Text>
            <BarChart
              categories={timeCategories}
              series={[
                {
                  name: "Wall time",
                  data: timeMinutes,
                  tone: "info",
                },
              ]}
              horizontal
              height={360}
              valueSuffix=" min"
              showValues
            />
            <Text size="small" tone="tertiary">
              Source: clean-cache PAAS profile. Buckets reconcile exactly to
              72.449 minutes.
            </Text>
          </CardBody>
        </Card>

        <Card size="lg">
          <CardHeader trailing="481 completed">Typical successful config</CardHeader>
          <CardBody>
            <Stack gap={14}>
              <Stat value="4.504s" label="Median process time" />
              <Stat value="16.507s" label="95th percentile process time" />
              <Divider />
              <Text>
                Median startup/import alone was <Text weight="semibold">2.562s</Text>.
              </Text>
              <Text>
                Median validation was <Text weight="semibold">0.991s</Text>.
              </Text>
              <Text>
                Median Triton timing was only{" "}
                <Text weight="semibold">17.5ms</Text>.
              </Text>
              <Text>
                Median shutdown plus scheduler detection was{" "}
                <Text weight="semibold">0.870s</Text>.
              </Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <H2>Validation and compilation dominate successful candidates</H2>
      <Text tone="secondary">
        The candidate and reference launch stages include compilation or
        loading compiled code. Kernel execution itself is measured in
        microseconds.
      </Text>
      <BarChart
        categories={validationCategories}
        series={[
          {
            name: "Wall time",
            data: validationMinutes,
            tone: "warning",
          },
        ]}
        horizontal
        height={245}
        valueSuffix=" min"
        showValues
      />
      <Text size="small" tone="tertiary">
        Category axis: validation stage · Value axis: summed wall time
        (minutes) · Source: 480 completed numerical validations.
      </Text>

      <Divider />

      <H2>Large XBLOCK values concentrate the cost</H2>
      <Text tone="secondary">
        All 19 timeouts occurred at XBLOCK 2048, 4096, or 8192. Each timeout
        cost about 62 seconds after termination grace.
      </Text>
      <BarChart
        categories={xblockCategories}
        series={[
          {
            name: "Wall time",
            data: xblockMinutes,
            tone: "danger",
          },
        ]}
        height={300}
        valueSuffix=" min"
        showValues={false}
      />
      <Text size="small" tone="tertiary">
        Category axis: XBLOCK · Value axis: total wall time (minutes) · Source:
        all 500 sweep configurations.
      </Text>
      <BarChart
        categories={xblockCategories}
        series={[
          {
            name: "Timeout count",
            data: xblockTimeouts,
            tone: "danger",
          },
        ]}
        height={220}
        valueSuffix=" timeouts"
        showValues
      />
      <Text size="small" tone="tertiary">
        Category axis: XBLOCK · Value axis: timed-out configurations (count) ·
        Source: 19 timeout records.
      </Text>

      <Divider />

      <H2>Optimization opportunities</H2>
      <BarChart
        categories={savingsCategories}
        series={[
          {
            name: "Potential wall-time reduction",
            data: savingsMinutes,
            tone: "success",
          },
        ]}
        horizontal
        height={245}
        valueSuffix=" min"
        showValues
      />
      <Text size="small" tone="tertiary">
        Category axis: proposed change · Value axis: potential reduction
        (minutes). The 16K cap is measured; 8-GPU sharding is an idealized
        schedule; polling is an upper bound.
      </Text>

      <Grid columns="repeat(auto-fit, minmax(300px, 1fr))" gap={18}>
        <Stack gap={8}>
          <H3>Highest-confidence change</H3>
          <Text>
            A 16,384 reduction block-product cap would skip 108
            configurations that consumed <Text weight="semibold">38m 30.9s</Text>,
            including every timeout.
          </Text>
          <Text>
            The winning configuration has block product 1,024 and remains in
            the capped search.
          </Text>
        </Stack>
        <Stack gap={8}>
          <H3>Largest wall-clock opportunity</H3>
          <Text>
            PAAS uses eight GPUs for different kernel files, but cannot split
            one kernel's configurations across them.
          </Text>
          <Text>
            An idealized schedule of these measured tasks finishes in about{" "}
            <Text weight="semibold">10m 6.8s</Text> before contention.
          </Text>
        </Stack>
      </Grid>

      <Callout tone="warning" title="Keep the correctness boundary explicit">
        Persistent workers or cached numerical references could remove large
        startup and reference costs, but they would no longer provide one fresh
        process and fresh reference per candidate. Treat those as separate
        validation-policy experiments.
      </Callout>

      <Text size="small" tone="tertiary">
        Cache state: both dedicated caches were absent at run start. The final
        Triton cache contained 3,921 files / 207.0 MiB; the standalone run did
        not populate the Inductor cache.
      </Text>
    </Stack>
  );
}
