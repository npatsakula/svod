//! Unified parallel execution for heterogeneous devices.
//!
//! The `UnifiedExecutor` handles kernel execution across any mix of devices
//! (CPU, CUDA, Metal, etc.) with proper synchronization and dependency tracking.
//!
//! # Design Principles
//!
//! 1. **Single abstraction** - One executor handles any device mix
//! 2. **Device-agnostic sync** - Timeline signals abstract over device-specific primitives
//! 3. **Zero overhead for single-device** - Fast path skips synchronization when possible
//! 4. **Buffer dependency tracking** - Following Tinygrad's `_access_resources()` pattern
//!
//! # Example
//!
//! ```ignore
//! let mut executor = UnifiedExecutor::new();
//! executor.add_device(DeviceSpec::Cpu)?;
//!
//! // Execute schedule - handles dependencies automatically
//! let output_id = executor.execute(&schedule)?;
//! ```
//!
//! # Execution Graph
//!
//! For complex schedules with multiple devices, the executor builds an execution
//! graph (DAG) where nodes are kernel operations and edges are buffer dependencies.
//! Independent kernels on the same device can be batched, and kernels on different
//! devices can run in parallel (with appropriate synchronization).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use svod_device::device::Device;
use svod_device::registry::DeviceRegistry;
use svod_device::{Allocator, Buffer, BufferId, CpuTimelineSignal, TimelineSignal};
use svod_dtype::DeviceSpec;

use crate::error::Result;

/// Per-device execution context.
///
/// Each device has its own timeline signal, queue, and allocator.
/// This enables parallel execution across devices with proper synchronization.
pub struct DeviceContext {
    /// Device specification (CPU, CUDA:0, etc.).
    pub device: DeviceSpec,
    /// Device abstraction for rendering/compiling/executing.
    pub device_handle: Arc<Device>,
    /// Timeline signal for this device's operations.
    pub signal: Arc<dyn TimelineSignal>,
    /// Current timeline value (monotonically increasing).
    pub timeline: AtomicU64,
    /// Allocator for this device.
    pub allocator: Arc<dyn Allocator>,
}

impl std::fmt::Debug for DeviceContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceContext")
            .field("device", &self.device)
            .field("timeline", &self.timeline.load(Ordering::Relaxed))
            .finish()
    }
}

impl DeviceContext {
    /// Create a new device context.
    pub fn new(device: Arc<Device>, signal: Arc<dyn TimelineSignal>) -> Self {
        let allocator = device.allocator.clone();
        let device_spec = device.device.clone();
        Self { device: device_spec, device_handle: device, signal, timeline: AtomicU64::new(0), allocator }
    }

    /// Get the next timeline value and increment.
    pub fn next_timeline(&self) -> u64 {
        self.timeline.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Get the current timeline value.
    pub fn current_timeline(&self) -> u64 {
        self.timeline.load(Ordering::Relaxed)
    }

    /// Signal that operations up to the given timeline value are complete.
    pub fn signal_completion(&self, value: u64) {
        self.signal.set(value);
    }

    /// Wait for operations up to the given timeline value to complete.
    pub fn wait_for(&self, value: u64) -> Result<()> {
        self.signal.wait(value, 0)?;
        Ok(())
    }
}

/// Cross-device synchronization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStrategy {
    /// Same device - no synchronization needed (operations are ordered).
    None,
    /// Same device type, different instance (e.g., CUDA:0 → CUDA:1).
    /// Use peer-to-peer events if available.
    PeerToPeer,
    /// Different device types (e.g., CUDA → CPU).
    /// Use CPU-mediated polling.
    CpuMediated,
}

/// A node in the execution graph representing a kernel or transfer operation.
#[derive(Debug, Clone)]
pub struct ExecutionNode {
    /// Unique identifier for this node (typically the kernel AST ID).
    pub id: u64,
    /// Device this operation executes on.
    pub device: DeviceSpec,
    /// Buffer IDs read by this operation.
    pub inputs: Vec<BufferId>,
    /// Buffer IDs written by this operation.
    pub outputs: Vec<BufferId>,
    /// IDs of nodes that must complete before this one (dependencies).
    pub predecessors: Vec<u64>,
    /// Whether this is a data transfer (COPY) or a computational kernel.
    pub is_transfer: bool,
    /// Buffer access information for parallel execution.
    /// Contains the full buffer list and output indices for dependency tracking.
    pub buffer_access: Option<KernelBufferAccess>,
}

/// Buffer access information for kernel scheduling.
///
/// Captures which buffers a kernel reads/writes; used by the executor and
/// memory planner to compute dependencies between operations.
#[derive(Debug, Clone)]
pub struct KernelBufferAccess {
    /// All buffer IDs accessed by this kernel (inputs and outputs).
    pub buffers: Vec<BufferId>,
    /// Indices into `buffers` that are outputs (written by the kernel).
    /// Other indices are inputs (read-only).
    pub output_indices: Vec<usize>,
}

/// Execution graph representing a DAG of kernel operations.
///
/// The graph is built from a schedule and captures buffer dependencies
/// between kernels. Independent kernels can be executed in parallel.
#[derive(Debug, Default)]
pub struct ExecutionGraph {
    /// Nodes in the graph, indexed by ID.
    nodes: HashMap<u64, ExecutionNode>,
    /// Execution order (topologically sorted node IDs).
    execution_order: Vec<u64>,
    /// Nodes grouped by device for batched execution.
    device_groups: HashMap<DeviceSpec, Vec<u64>>,
}

impl ExecutionGraph {
    /// Create a new empty execution graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: ExecutionNode) {
        let id = node.id;
        let device = node.device.clone();
        self.nodes.insert(id, node);
        self.device_groups.entry(device).or_default().push(id);
    }

    /// Get a node by ID.
    pub fn node(&self, id: u64) -> Option<&ExecutionNode> {
        self.nodes.get(&id)
    }

    /// Get all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &ExecutionNode> {
        self.nodes.values()
    }

    /// Compute topological order and find parallelizable groups.
    ///
    /// Returns groups of nodes that can be executed in parallel.
    /// Each group contains nodes with no dependencies on each other.
    pub fn compute_parallel_groups(&mut self) -> Vec<Vec<u64>> {
        // Build in-degree map. Predecessor lists are deduplicated via
        // sort-and-dedup on a local SmallVec so a node that lists the same
        // predecessor twice doesn't decrement in-degree more than once during
        // Kahn's traversal. Successor sets remain HashSet to keep insertion
        // O(1) when many nodes share predecessors.
        let mut in_degree: HashMap<u64, usize> = HashMap::new();
        let mut successors: HashMap<u64, HashSet<u64>> = HashMap::new();

        for node in self.nodes.values() {
            in_degree.entry(node.id).or_insert(0);
            let mut preds: smallvec::SmallVec<[u64; 8]> = node.predecessors.iter().copied().collect();
            preds.sort_unstable();
            preds.dedup();
            for &pred in &preds {
                successors.entry(pred).or_default().insert(node.id);
                *in_degree.entry(node.id).or_insert(0) += 1;
            }
        }

        // Kahn's algorithm with level grouping
        let mut groups = Vec::new();
        let mut ready: Vec<u64> = in_degree.iter().filter(|&(_, deg)| *deg == 0).map(|(&id, _)| id).collect();

        while !ready.is_empty() {
            // All nodes in ready can be executed in parallel
            groups.push(ready.clone());

            // Record execution order
            self.execution_order.extend(ready.iter().copied());

            // Find next batch
            let mut next_ready = Vec::new();
            for id in ready {
                if let Some(succs) = successors.get(&id) {
                    for &succ in succs {
                        let deg = in_degree.get_mut(&succ).unwrap();
                        *deg -= 1;
                        if *deg == 0 {
                            next_ready.push(succ);
                        }
                    }
                }
            }
            ready = next_ready;
        }

        groups
    }

    /// Get nodes grouped by device.
    pub fn device_groups(&self) -> &HashMap<DeviceSpec, Vec<u64>> {
        &self.device_groups
    }

    /// Check if all nodes have been visited (no cycles).
    pub fn is_valid(&self) -> bool {
        self.execution_order.len() == self.nodes.len()
    }
}

/// Unified executor for heterogeneous device execution.
///
/// Manages device contexts to enable parallel execution across any mix of devices.
/// Uses timeline signals for cross-device synchronization.
///
/// # Stateless Execution Model (Tinygrad-Aligned)
///
/// The executor follows Tinygrad's stateless execution model where:
/// - Dependencies are computed at schedule time, not runtime
/// - ExecutionPlan pre-computes kernel order via topological sort
/// - No runtime dependency tracking is needed (zero memory accumulation)
/// - Timeline signals handle cross-device synchronization only
pub struct UnifiedExecutor {
    /// Per-device execution contexts.
    contexts: HashMap<DeviceSpec, DeviceContext>,

    /// Device registry for looking up allocators.
    registry: &'static DeviceRegistry,
}

impl std::fmt::Debug for UnifiedExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedExecutor").field("contexts", &self.contexts.keys().collect::<Vec<_>>()).finish()
    }
}

impl UnifiedExecutor {
    /// Create a new unified executor.
    pub fn new(registry: &'static DeviceRegistry) -> Self {
        Self { contexts: HashMap::new(), registry }
    }

    /// Add a device to the executor.
    ///
    /// Creates the device context with timeline signal and queues.
    pub fn add_device(&mut self, device_spec: DeviceSpec) -> Result<()> {
        if self.contexts.contains_key(&device_spec) {
            return Ok(()); // Already added
        }

        // Create device handle
        let device = crate::DEVICE_FACTORIES.device(&device_spec, self.registry)?;

        // Executor-level ordering is a host signal for every backend; GPU
        // backends order their own submissions on the device beneath it.
        let signal: Arc<dyn TimelineSignal> = Arc::new(CpuTimelineSignal::new());

        let ctx = DeviceContext::new(device, signal);
        self.contexts.insert(device_spec, ctx);

        Ok(())
    }

    /// Get the device context for a device specification.
    pub fn context(&self, device: &DeviceSpec) -> Option<&DeviceContext> {
        self.contexts.get(device)
    }

    /// Get the device context mutably.
    pub fn context_mut(&mut self, device: &DeviceSpec) -> Option<&mut DeviceContext> {
        self.contexts.get_mut(device)
    }

    /// Determine the synchronization strategy between two devices.
    pub fn sync_strategy(from: &DeviceSpec, to: &DeviceSpec) -> SyncStrategy {
        if from == to {
            SyncStrategy::None
        } else if std::mem::discriminant(from) == std::mem::discriminant(to) {
            // Same device type (e.g., both CUDA)
            SyncStrategy::PeerToPeer
        } else {
            // Different device types
            SyncStrategy::CpuMediated
        }
    }

    /// Check if all operations on a single device.
    ///
    /// Returns `Some(device)` if all buffers are on the same device,
    /// enabling the fast single-device path.
    pub fn single_device_check(&self, buffers: &[&Buffer]) -> Option<DeviceSpec> {
        if buffers.is_empty() {
            return None;
        }

        let first_device = buffers[0].allocator().device_spec();

        for buffer in buffers.iter().skip(1) {
            if buffer.allocator().device_spec() != first_device {
                return None;
            }
        }

        Some(first_device)
    }

    /// Synchronize all devices.
    ///
    /// Waits for all pending operations to complete on all devices.
    pub fn synchronize_all(&self) -> Result<()> {
        for ctx in self.contexts.values() {
            let current = ctx.current_timeline();
            if current > 0 {
                ctx.wait_for(current)?;
            }
        }
        Ok(())
    }

    /// Execute a kernel (sequential execution).
    ///
    /// ExecutionPlan pre-computes kernel order at schedule time, so no runtime
    /// dependency tracking is needed. This follows Tinygrad's stateless execution model.
    ///
    /// # Arguments
    ///
    /// * `device` - Device to execute on
    /// * `execute_fn` - Function that performs the actual kernel execution
    ///
    /// # Returns
    ///
    /// The timeline value for this execution (can be used for cross-device sync).
    pub fn execute_kernel<F>(&mut self, device: &DeviceSpec, execute_fn: F) -> Result<u64>
    where
        F: FnOnce() -> Result<()>,
    {
        // 1. Ensure device context exists
        if !self.contexts.contains_key(device) {
            self.add_device(device.clone())?;
        }

        // 2. Get next timeline value for this execution
        let timeline = self.contexts.get(device).unwrap().next_timeline();

        // 3. Execute the kernel
        execute_fn()?;

        // 4. Signal completion (for cross-device synchronization)
        if let Some(ctx) = self.contexts.get(device) {
            ctx.signal_completion(timeline);
        }

        Ok(timeline)
    }

    /// Execute a buffer transfer (COPY operation).
    ///
    /// Handles cross-device transfers with appropriate synchronization:
    /// - Same device: Direct copy using device's copy queue
    /// - Same vendor (e.g., CUDA:0 → CUDA:1): Peer-to-peer transfer
    /// - Different vendors (e.g., CUDA → CPU): Stage through host memory
    ///
    /// # Arguments
    ///
    /// * `src` - Source buffer
    /// * `dst` - Destination buffer (must be pre-allocated)
    /// * `src_device` - Device the source buffer is on
    /// * `dst_device` - Device the destination buffer is on
    ///
    /// # Returns
    ///
    /// The timeline value for this transfer operation.
    pub fn execute_transfer(
        &mut self,
        src: &Buffer,
        dst: &mut Buffer,
        src_device: &DeviceSpec,
        dst_device: &DeviceSpec,
    ) -> Result<u64> {
        // Ensure both device contexts exist
        if !self.contexts.contains_key(src_device) {
            self.add_device(src_device.clone())?;
        }
        if !self.contexts.contains_key(dst_device) {
            self.add_device(dst_device.clone())?;
        }

        // Get timeline for destination device (where the result will be used)
        let timeline = self.contexts.get(dst_device).unwrap().next_timeline();

        // Perform the transfer based on sync strategy
        match Self::sync_strategy(src_device, dst_device) {
            SyncStrategy::None => {
                // Same device - direct copy
                dst.copy_from(src)?;
            }
            SyncStrategy::PeerToPeer => {
                // Same vendor (e.g., both CUDA) - use peer-to-peer if available
                // For now, fall back to copy_from which handles this
                dst.copy_from(src)?;
            }
            SyncStrategy::CpuMediated => {
                // Different vendors - stage through CPU
                // First, wait for source device operations to complete
                if let Some(src_ctx) = self.contexts.get(src_device) {
                    let src_timeline = src_ctx.current_timeline();
                    if src_timeline > 0 {
                        src_ctx.wait_for(src_timeline)?;
                    }
                }

                // copy_from handles the staging internally for cross-device copies
                dst.copy_from(src)?;

                // Wait for destination device operations if needed
                if let Some(dst_ctx) = self.contexts.get(dst_device) {
                    let dst_timeline = dst_ctx.current_timeline();
                    if dst_timeline > 0 {
                        dst_ctx.wait_for(dst_timeline)?;
                    }
                }
            }
        }

        // Signal completion (for cross-device synchronization)
        if let Some(ctx) = self.contexts.get(dst_device) {
            ctx.signal_completion(timeline);
        }

        Ok(timeline)
    }
}

/// Global executor instance.
///
/// For most use cases, a single global executor is sufficient.
/// Thread-safety is handled by timeline signals and dependency tracking.
static EXECUTOR: once_cell::sync::Lazy<parking_lot::Mutex<UnifiedExecutor>> =
    once_cell::sync::Lazy::new(|| parking_lot::Mutex::new(UnifiedExecutor::new(svod_device::registry::registry())));

/// Get access to the global executor.
pub fn global_executor() -> parking_lot::MutexGuard<'static, UnifiedExecutor> {
    EXECUTOR.lock()
}

#[cfg(test)]
#[path = "test/unit/executor.rs"]
mod tests;
