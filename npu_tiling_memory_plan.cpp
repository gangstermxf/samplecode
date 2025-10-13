// NPU Tiling and Memory Allocation Project Plan and Schedule (C++)
// This file encodes a project plan, schedule, and skeletal APIs for an
// NPU tiling + memory allocation system that takes an AI model graph
// (operations + shapes) and targets a memory hierarchy with L1/L2/DDR.
// L1 supports a Y-dimension ring buffer, and DMA can move data among
// L1/L2/DDR in parallel with compute.

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <sstream>

// -----------------------------
// Core domain models
// -----------------------------

enum class MemoryType { L1_SRAM, L2_SRAM, DDR };

struct TensorShape {
    int n {1};
    int c {1};
    int h {1};
    int w {1};
};

struct Operation {
    std::string name;               // e.g. "conv2d_1"
    std::string opType;             // e.g. "Conv2D", "Gemm", "Relu"
    std::vector<TensorShape> inputs;
    std::vector<TensorShape> outputs;
    std::map<std::string, std::string> attributes; // kernel, stride, pad, groups, etc.
};

struct ModelGraph {
    std::vector<Operation> operations; // topologically sorted preferred
};

struct RingBufferPolicy {
    bool enabled {true};          // L1 ring buffer enabled
    std::string dimension {"Y"}; // ring along Y (height) dimension
    int minLines {0};             // minimal lines required (e.g., kernel_h or kernel_h + (pipeline-1)*stride)
};

struct DmaCapability {
    bool supportsOverlapWithCompute {true};
    int maxChannels {2}; // number of concurrent DMA channels
    bool l1ToL2 {true};
    bool l2ToL1 {true};
    bool ddrToL2 {true};
    bool l2ToDdr {true};
    bool ddrToL1 {true};
    bool l1ToDdr {true};
};

// -----------------------------
// Project planning models
// -----------------------------

struct PlanTask {
    std::string id;                // e.g., "T1"
    std::string title;             // short summary
    std::string description;       // details
    std::vector<std::string> dependsOn; // task IDs
    int estimateDays {0};          // ideal engineering days
    std::string deliverable;       // artifact/result
};

struct Milestone {
    std::string title;
    int dueWeek {0};
    std::vector<std::string> taskIds;
    std::string acceptanceCriteria;
};

struct ScheduleWeek {
    int week {0};
    std::vector<std::string> focusAreas; // themes
    std::vector<std::string> taskIds;    // tasks planned
};

struct ProjectPlan {
    std::string projectName;
    std::vector<std::string> assumptions;
    ModelGraph inputExpectation; // illustrative only
    RingBufferPolicy ringPolicy;
    DmaCapability dma;
    std::vector<PlanTask> tasks;
    std::vector<Milestone> milestones;
    std::vector<ScheduleWeek> schedule;
};

// -----------------------------
// Skeletal public API targets (to be implemented during the project)
// -----------------------------

namespace npu {

enum class DataLayout { NCHW, NHWC };

struct Buffer {
    MemoryType memoryType {MemoryType::DDR};
    size_t bytes {0};
    bool isRingBuffer {false}; // true when mapped to L1 Y-ring
};

struct TileShape { int n{1}, c{1}, h{1}, w{1}; };

struct TilingDecision {
    TileShape inputTile;
    TileShape outputTile;
    int yRingLines {0};     // number of Y lines to keep in L1
    bool useDoubleBuffer {true}; // double/triple buffering for overlap
};

class TilingStrategyEngine {
public:
    TilingDecision chooseBest(const Operation& op,
                              const TensorShape& input,
                              const TensorShape& output,
                              const RingBufferPolicy& ring,
                              const DmaCapability& dma) const;
};

class MemoryAllocator {
public:
    // Assigns buffers across L1/L2/DDR, with possible spills and ring placement
    std::map<std::string, Buffer> allocate(const ModelGraph& graph,
                                           const std::vector<TilingDecision>& tilingDecisions) const;
};

class DmaPlanner {
public:
    // Builds a DMA schedule overlapping copy with compute, honoring channels and deps
    void buildOverlapPlan(const ModelGraph& graph,
                          const std::vector<TilingDecision>& tilingDecisions,
                          const DmaCapability& dma) const;
};

class PipelineScheduler {
public:
    // Produces an interleaved compute + DMA pipeline with ring-buffered L1
    void buildPipelinedSchedule(const ModelGraph& graph,
                                const std::vector<TilingDecision>& tilingDecisions,
                                const DmaCapability& dma,
                                const RingBufferPolicy& ring) const;
};

} // namespace npu

// -----------------------------
// Plan construction
// -----------------------------

static PlanTask T(
    const std::string& id,
    const std::string& title,
    const std::string& desc,
    std::vector<std::string> deps,
    int days,
    const std::string& deliverable
) {
    return PlanTask{ id, title, desc, std::move(deps), days, deliverable };
}

ProjectPlan buildPlan() {
    ProjectPlan plan;
    plan.projectName = "NPU Tiling and Memory Allocation";
    plan.assumptions = {
        "Input is an AI model graph whose nodes are operations with static shapes.",
        "Memory hierarchy: L1 SRAM, L2 SRAM, DDR; L1 is size-limited and fastest.",
        "L1 SRAM supports a Y-dimension ring buffer for streaming rows.",
        "DMA can move data among L1/L2/DDR overlapping compute; up to 2 channels.",
        "Primary target ops: Conv2D, DepthwiseConv2D, Gemm/MatMul, Pool, Elementwise.",
        "First release focuses on inference graphs with static shapes.",
    };

    plan.ringPolicy = { true, "Y", /*minLines*/ 0 };
    plan.dma = { true, /*maxChannels*/ 2, true, true, true, true, true, true };

    // Tasks (10 weeks baseline plan)
    plan.tasks = {
        T("T1",  "Define IR and graph import",
          "Design a minimal IR: ops, tensors, attributes, edges; implement ONNX-like importer; preserve shapes.",
          {}, 3, "IR headers + importer that builds ModelGraph"),
        T("T2",  "Static shape and layout inference",
          "Propagate tensor shapes/layouts (NCHW/NHWC) and validate graph consistency.",
          {"T1"}, 2, "Shape inference pass with unit tests"),
        T("T3",  "Memory model and size accounting",
          "Define L1/L2/DDR capacities and bandwidths; implement byte-accurate tensor size and tile size accounting.",
          {"T1"}, 2, "MemoryConfig + size calculators"),
        T("T4",  "Liveness and buffer reuse analysis",
          "Compute tensor live ranges; enable buffer reuse to reduce peak memory; build interference graph.",
          {"T2","T3"}, 3, "Liveness pass + reuse assignment (greedy)"),
        T("T5",  "Cost model v0",
          "Latency estimate for compute and DMA; model overlap; inputs: tile shape, memory level, bandwidths.",
          {"T3"}, 3, "CostModel::estimate(tile) API"),
        T("T6",  "Single-op tiling search (Conv/Gemm)",
          "Enumerate tile shapes within L1/L2; respect alignment; evaluate with cost model; select best.",
          {"T5"}, 4, "TilingStrategyEngine for Conv2D/Gemm"),
        T("T7",  "Y-dimension L1 ring buffer policy",
          "Derive minimal Y-lines for kernel/stride; plan ring slots; double/triple buffering to cover DMA latency.",
          {"T6"}, 3, "RingBufferPolicy integration and estimates"),
        T("T8",  "DMA overlap scheduler",
          "Generate prefetch/writeback schedule using available channels; enforce deps; avoid deadlocks.",
          {"T5","T6"}, 4, "DmaPlanner with overlap checks"),
        T("T9",  "Memory assignment across L1/L2/DDR with spill",
          "Allocate buffers to L1 if resident, else L2, else DDR; plan spills and prefetch windows.",
          {"T4","T6","T8"}, 4, "MemoryAllocator v1 with spill support"),
        T("T10", "Pipeline schedule builder",
          "Interleave compute + DMA to form a pipeline; ring-buffered Y iteration; stabilize order.",
          {"T6","T7","T8","T9"}, 4, "PipelineScheduler v1"),
        T("T11", "DMA micro-scheduler and channel allocation",
          "Pack transfers into bursts; assign channels; minimize gaps; respect SRAM port limits.",
          {"T8"}, 3, "Improved DmaPlanner channel-aware"),
        T("T12", "Tile sizing heuristics v1",
          "Fast approximations for search warm-start; tie-break by reuse and alignment.",
          {"T6"}, 2, "Heuristic seeding in TilingStrategyEngine"),
        T("T13", "Graph-level tiling propagation",
          "Propagate producer/consumer tiling to reduce reformat copies; fuse simple ops.",
          {"T6","T9"}, 4, "Producer-consumer tiling propagation pass"),
        T("T14", "Schedule simulator and validator",
          "Simulate pipeline timeline (compute + DMA); detect over-commit of L1; verify deps.",
          {"T10","T11"}, 4, "Deterministic simulator + assertions"),
        T("T15", "Hardware calibration",
          "Measure/ingest NPU latency/bandwidth; tune cost model; set default ring sizes.",
          {"T5"}, 3, "Calibrated CostModel params"),
        T("T16", "Integration tests on toy graphs",
          "Conv blocks, residuals, small CNNs; assert schedule validity and memory limits.",
          {"T10","T14"}, 3, "End-to-end tests"),
        T("T17", "Performance regression harness",
          "Benchmark suite + golden latencies; track overlap efficiency and spills.",
          {"T15","T16"}, 3, "Perf harness + baselines"),
        T("T18", "Operator coverage: Depthwise, Pool, Eltwise",
          "Extend tiling/allocator to depthwise conv, pooling, elementwise ops.",
          {"T6","T9"}, 4, "Extended op coverage"),
        T("T19", "Documentation and examples",
          "User guide, API docs, examples; describe ring buffer and DMA overlap.",
          {"T10","T16"}, 3, "Docs + sample graphs"),
        T("T20", "Release v1",
          "Stabilize APIs; tag release; publish artifacts.",
          {"T17","T19"}, 2, "v1.0 tag + release notes"),
    };

    plan.milestones = {
        { "M1: IR + Shapes ready", 2, {"T1","T2","T3"},
          "Can import graphs, infer shapes, compute sizes." },
        { "M2: Single-op tiling + ring buffer + overlap", 4, {"T5","T6","T7","T8"},
          "Select tiles for Conv/Gemm, simulate DMA overlap with Y-ring." },
        { "M3: End-to-end pipeline", 6, {"T9","T10","T14"},
          "Build valid pipelined schedules on toy graphs." },
        { "M4: Graph tiling propagation + spills", 8, {"T11","T12","T13"},
          "Producer-consumer tiling, basic spills, improved DMA packing." },
        { "M5: Calibrated perf + docs + release", 10, {"T15","T16","T17","T18","T19","T20"},
          "Perf baselines established; docs published; v1 released." },
    };

    plan.schedule = {
        { 1,  {"IR & Shapes", "Memory accounting"}, {"T1","T2","T3"} },
        { 2,  {"Liveness", "Cost model v0"}, {"T4","T5"} },
        { 3,  {"Single-op tiling"}, {"T6"} },
        { 4,  {"Y-ring policy", "DMA overlap"}, {"T7","T8"} },
        { 5,  {"Allocator with spill"}, {"T9"} },
        { 6,  {"Pipeline builder", "Simulator"}, {"T10","T14"} },
        { 7,  {"DMA micro-sched", "Heuristics"}, {"T11","T12"} },
        { 8,  {"Graph tiling propagation"}, {"T13"} },
        { 9,  {"HW calibration", "Integration tests"}, {"T15","T16"} },
        { 10, {"Perf harness", "Op coverage", "Docs", "Release"}, {"T17","T18","T19","T20"} },
    };

    return plan;
}

// -----------------------------
// Rendering helpers
// -----------------------------

std::string renderPlan(const ProjectPlan& p) {
    std::ostringstream out;

    out << "Project: " << p.projectName << "\n\n";

    out << "Assumptions:\n";
    for (const auto& a : p.assumptions) {
        out << "  - " << a << "\n";
    }
    out << "\nMemory Hierarchy:\n"
        << "  - L1 SRAM: fastest, limited capacity; supports Y-dim ring buffer.\n"
        << "  - L2 SRAM: medium capacity/latency; staging for DDR <-> L1.\n"
        << "  - DDR: largest capacity; highest latency.\n\n";

    out << "DMA Capability:\n";
    out << "  - Overlap with compute: " << (p.dma.supportsOverlapWithCompute ? "yes" : "no") << "\n";
    out << "  - Channels: " << p.dma.maxChannels << "\n\n";

    out << "L1 Y-Ring Buffer Policy:\n";
    out << "  - Enabled: " << (p.ringPolicy.enabled ? "yes" : "no") << ", dimension: " << p.ringPolicy.dimension
        << ", minLines: " << p.ringPolicy.minLines << " (derived per op).\n\n";

    out << "Work Breakdown (Tasks):\n";
    for (const auto& t : p.tasks) {
        out << "  * " << t.id << " - " << t.title << " (" << t.estimateDays << "d)\n"
            << "    Desc: " << t.description << "\n"
            << "    Deps: ";
        if (t.dependsOn.empty()) out << "-";
        for (size_t i = 0; i < t.dependsOn.size(); ++i) {
            out << t.dependsOn[i] << (i + 1 < t.dependsOn.size() ? ", " : "");
        }
        out << "\n    Deliverable: " << t.deliverable << "\n";
    }
    out << "\nMilestones:\n";
    for (const auto& m : p.milestones) {
        out << "  - " << m.title << " (Week " << m.dueWeek << ")\n"
            << "    Includes: ";
        for (size_t i = 0; i < m.taskIds.size(); ++i) {
            out << m.taskIds[i] << (i + 1 < m.taskIds.size() ? ", " : "");
        }
        out << "\n    Acceptance: " << m.acceptanceCriteria << "\n";
    }

    out << "\nWeek-by-Week Schedule (T0 = project start):\n";
    for (const auto& w : p.schedule) {
        out << "  Week " << w.week << ":\n";
        out << "    Focus: ";
        for (size_t i = 0; i < w.focusAreas.size(); ++i) {
            out << w.focusAreas[i] << (i + 1 < w.focusAreas.size() ? ", " : "");
        }
        out << "\n    Tasks: ";
        for (size_t i = 0; i < w.taskIds.size(); ++i) {
            out << w.taskIds[i] << (i + 1 < w.taskIds.size() ? ", " : "");
        }
        out << "\n";
    }

    out << "\nTechnical Notes (Key Design Details):\n";
    out << "  - Input graph: Each op has typed inputs/outputs and static shapes.\n";
    out << "  - Tiling: Enumerate (n,c,h,w) tiles that fit in L1/L2; honor alignment.\n";
    out << "  - Y-ring: For Conv2D with kernel_h and stride_h, min ring lines typically\n"
           "    = kernel_h + (pipeline_depth - 1) * stride_h. Use double/triple buffering\n"
           "    to overlap DMA prefetch of next lines with compute on current lines.\n";
    out << "  - Memory assignment: Prefer L1 residency for hot tiles; spill to L2/DDR as needed;\n"
           "    prefetch windows sized from cost model.\n";
    out << "  - Overlap: Maintain dependency graph so that DMA reads/writes interleave with\n"
           "    compute; avoid port conflicts and channel overuse; enforce barriers only\n"
           "    where required.\n";
    out << "  - Validation: Simulator verifies no L1/L2 capacity overflow across timeline;\n"
           "    checks for deadlocks and unmet deps.\n";

    out << "\nTesting & Perf:\n";
    out << "  - Unit tests: IR, shape inference, liveness, allocator, tiling selection.\n";
    out << "  - Integration: Small CNNs and residual blocks with ring-buffered convs.\n";
    out << "  - Perf: Report overlap ratio (DMA hidden), L1 hit rate, spill bytes,\n"
           "    estimated latency vs calibrated model.\n";

    out << "\nDeliverables by Release v1:\n";
    out << "  - Import + IR + shape/liveness passes.\n";
    out << "  - Tiling strategy for Conv/Gemm with Y-ring buffering.\n";
    out << "  - Memory allocator across L1/L2/DDR with spills.\n";
    out << "  - DMA overlap planner and pipelined schedule.\n";
    out << "  - Simulator/validator, tests, docs, and sample models.\n";

    return out.str();
}

int main() {
    ProjectPlan plan = buildPlan();
    std::cout << renderPlan(plan) << std::endl;
    return 0;
}
