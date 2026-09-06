//! A captured kernel chain as a CUDA graph. Nodes depend on the producers the
//! host hazard analysis recorded (`GraphKernel::deps`), so independent
//! kernels may overlap; a replay whose buffer bindings alias differently from
//! the captured ones (which is what those hazards were computed on) falls
//! back to a capture-order chain, and profiled replays run that chain with
//! event-record nodes around every kernel.

use std::sync::Arc;

use parking_lot::Mutex;

use super::device::{CudaDevice, CudaEvent, CudaStream};
use super::program::{CudaModule, CudaProgram, Launch, extra_array};
use super::sync::{CudaCompletionToken, CudaDispatchTimestamps};
use super::sys::{CUfunction, CUgraph, CUgraphExec, CUgraphNode, CudaKernelNodeParams};
use crate::device::{Graph, GraphKernel, Program};
use crate::hcq::ClikeKernargLayout;
use crate::sync::{CompletionToken, DispatchTimestamps};
use crate::{Error, Result};

struct Node {
    name: String,
    _module: Arc<CudaModule>,
    function: CUfunction,
    launch: Launch,
    layout: ClikeKernargLayout,
    deps: Vec<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Topology {
    /// Edges from `GraphKernel::deps`.
    Dag,
    /// Every kernel after the previous one.
    Chain,
}

/// One instantiated executable graph over the shared nodes.
struct Exec {
    dev: Arc<CudaDevice>,
    graph: CUgraph,
    exec: CUgraphExec,
    kernels: Vec<CUgraphNode>,
    /// `(start, end)` event-record nodes per kernel; empty when unprofiled.
    stamps: Vec<(CUgraphNode, CUgraphNode)>,
    events: Vec<(Arc<CudaEvent>, Arc<CudaEvent>)>,
    /// The bindings its kernel nodes currently hold.
    packed: (Vec<u64>, Vec<i64>),
}

impl Drop for Exec {
    fn drop(&mut self) {
        if self.dev.enter().is_err() {
            return;
        }
        let api = self.dev.api();
        // SAFETY: handles this value created; the driver defers destruction
        // of a launched exec until it retires.
        unsafe {
            (api.graph_exec_destroy)(self.exec);
            (api.graph_destroy)(self.graph);
        }
    }
}

struct State {
    blobs: Vec<Vec<u8>>,
    dag: Exec,
    chain: Option<Exec>,
    profiled: Option<Exec>,
    last: Option<CudaCompletionToken>,
}

pub struct CudaGraph {
    dev: Arc<CudaDevice>,
    stream: CudaStream,
    nodes: Vec<Node>,
    captured: (Vec<u64>, Vec<i64>),
    /// Which flattened buffer slots alias at capture (see [`alias_signature`]).
    aliasing: Vec<usize>,
    state: Mutex<State>,
}

impl std::fmt::Debug for CudaGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaGraph")
            .field("kernels", &self.nodes.iter().map(|node| node.name.as_str()).collect::<Vec<_>>())
            .field("buffers", &self.captured.0.len())
            .field("vals", &self.captured.1.len())
            .finish_non_exhaustive()
    }
}

/// For every slot, the first slot bound to the same address: two bindings
/// with equal signatures have identical hazards, so capture-time
/// dependencies stay valid.
pub(crate) fn alias_signature(buffers: &[u64]) -> Vec<usize> {
    buffers
        .iter()
        .enumerate()
        .map(|(index, address)| buffers[..index].iter().position(|earlier| earlier == address).unwrap_or(index))
        .collect()
}

impl CudaGraph {
    /// Capture `kernels`; `Ok(None)` for chains this backend cannot graph
    /// (empty, non-CUDA programs, programs of another device).
    pub fn capture(dev: Arc<CudaDevice>, kernels: &[GraphKernel<'_>]) -> Result<Option<Box<dyn Graph>>> {
        if kernels.is_empty() {
            return Ok(None);
        }
        if let Some(error) = dev.poison_error() {
            return Err(error);
        }
        let mut nodes = Vec::with_capacity(kernels.len());
        let mut blobs = Vec::with_capacity(kernels.len());
        let (mut buffers, mut vals) = (Vec::new(), Vec::new());
        for (index, kernel) in kernels.iter().enumerate() {
            let Some(program) = kernel.program.as_any().downcast_ref::<CudaProgram>() else { return Ok(None) };
            if !Arc::ptr_eq(program.device(), &dev) {
                return Ok(None);
            }
            if let Some(&dep) = kernel.deps.iter().find(|&&dep| dep >= index) {
                return Err(Error::Runtime {
                    message: format!(
                        "graph kernel {index} ('{}') depends on kernel {dep}, which is not earlier",
                        program.name()
                    ),
                });
            }
            let addresses: Vec<u64> = kernel.buffers.iter().map(|pointer| *pointer as u64).collect();
            blobs.push(program.pack(&addresses, &kernel.vals)?);
            buffers.extend_from_slice(&addresses);
            vals.extend_from_slice(&kernel.vals);
            nodes.push(Node {
                name: program.name().to_string(),
                _module: Arc::clone(program.module()),
                function: program.function(),
                launch: program.launch_dims(kernel.global_size, kernel.local_size)?,
                layout: program.layout().clone(),
                deps: kernel.deps.clone(),
            });
        }
        let dag = Exec::build(&dev, &nodes, &mut blobs, Topology::Dag, false, &buffers, &vals)?;
        tracing::debug!(kernels = nodes.len(), buffers = buffers.len(), "captured CUDA graph");
        Ok(Some(Box::new(Self {
            stream: CudaStream::new(Arc::clone(&dev))?,
            dev,
            nodes,
            aliasing: alias_signature(&buffers),
            captured: (buffers, vals),
            state: Mutex::new(State { blobs, dag, chain: None, profiled: None, last: None }),
        })))
    }

    /// The effective bindings of a replay: empty slices replay the capture.
    fn bindings<'a>(&'a self, buffers: &'a [u64], vals: &'a [i64]) -> Result<(&'a [u64], &'a [i64])> {
        let (expected_buffers, expected_vals) = (self.captured.0.len(), self.captured.1.len());
        if (!buffers.is_empty() && buffers.len() != expected_buffers)
            || (!vals.is_empty() && vals.len() != expected_vals)
        {
            return Err(Error::ProgramAbiMismatch {
                reason: format!(
                    "CUDA graph replay expected {expected_buffers} buffers/{expected_vals} vars, got {}/{}",
                    buffers.len(),
                    vals.len()
                ),
            });
        }
        Ok((
            if buffers.is_empty() { &self.captured.0 } else { buffers },
            if vals.is_empty() { &self.captured.1 } else { vals },
        ))
    }

    /// Launch on this graph's lane, ordered after published copies, and
    /// return the token covering it; the launch stays unpublished until the
    /// token is recorded on its storages or waited.
    fn launch(&self, exec: &Exec) -> Result<CudaCompletionToken> {
        let api = self.dev.enter()?;
        let lane = self.stream.lane();
        self.dev.order_launch(lane)?;
        lane.mark_unpublished();
        // SAFETY: a live exec launched on this graph's stream.
        self.dev.check(unsafe { (api.graph_launch)(exec.exec, lane.raw()) }, "cuGraphLaunch")?;
        self.stream.token()
    }
}

impl Exec {
    /// Instantiate one graph over `nodes`, packed with `buffers`/`vals`.
    fn build(
        dev: &Arc<CudaDevice>,
        nodes: &[Node],
        blobs: &mut [Vec<u8>],
        topology: Topology,
        profile: bool,
        buffers: &[u64],
        vals: &[i64],
    ) -> Result<Self> {
        let api = dev.enter()?;
        let mut graph = CUgraph::NULL;
        // SAFETY: out-pointer to a live handle slot.
        unsafe { (api.graph_create)(&mut graph, 0) }.check("cuGraphCreate")?;
        let mut this = Self {
            dev: Arc::clone(dev),
            graph,
            exec: CUgraphExec::NULL,
            kernels: Vec::new(),
            stamps: Vec::new(),
            events: Vec::new(),
            packed: (Vec::new(), Vec::new()),
        };
        let mut tails: Vec<CUgraphNode> = Vec::with_capacity(nodes.len());
        let (mut buffer_offset, mut var_offset) = (0, 0);
        for (index, node) in nodes.iter().enumerate() {
            let slot_buffers = &buffers[buffer_offset..buffer_offset + node.layout.globals];
            let slot_vals = &vals[var_offset..var_offset + node.layout.vars];
            node.layout.pack(&mut blobs[index], slot_buffers, slot_vals)?;
            buffer_offset += node.layout.globals;
            var_offset += node.layout.vars;

            let mut deps: Vec<CUgraphNode> = match topology {
                Topology::Dag => node.deps.iter().map(|&dep| tails[dep]).collect(),
                Topology::Chain => tails.last().copied().into_iter().collect(),
            };
            let mut stamps = None;
            if profile {
                let start = Arc::new(CudaEvent::new(Arc::clone(dev), true)?);
                let end = Arc::new(CudaEvent::new(Arc::clone(dev), true)?);
                let start_node = this.add_event_node(&deps, &start)?;
                deps = vec![start_node];
                stamps = Some((start_node, Arc::clone(&end)));
                this.events.push((start, end));
            }
            let kernel = this.add_kernel_node(&deps, node, &mut blobs[index])?;
            this.kernels.push(kernel);
            let tail = match stamps {
                Some((start_node, end)) => {
                    let end_node = this.add_event_node(&[kernel], &end)?;
                    this.stamps.push((start_node, end_node));
                    end_node
                }
                None => kernel,
            };
            tails.push(tail);
        }
        // SAFETY: out-pointer to a live handle slot; the graph is complete.
        unsafe { (api.graph_instantiate_with_flags)(&mut this.exec, graph, 0) }.check("cuGraphInstantiate")?;
        this.packed = (buffers.to_vec(), vals.to_vec());
        Ok(this)
    }

    fn params(
        node: &Node,
        blob: &mut [u8],
        size: &mut usize,
        extra: &mut [*mut std::ffi::c_void; 5],
    ) -> CudaKernelNodeParams {
        *size = blob.len();
        *extra = extra_array(blob, size);
        let [gx, gy, gz] = node.launch.grid;
        let [bx, by, bz] = node.launch.block;
        CudaKernelNodeParams {
            func: node.function,
            grid_dim_x: gx,
            grid_dim_y: gy,
            grid_dim_z: gz,
            block_dim_x: bx,
            block_dim_y: by,
            block_dim_z: bz,
            shared_mem_bytes: 0,
            kernel_params: std::ptr::null_mut(),
            extra: extra.as_mut_ptr(),
            kern: super::sys::CUkernel::NULL,
            ctx: super::sys::CUcontext::NULL,
        }
    }

    fn add_kernel_node(&self, deps: &[CUgraphNode], node: &Node, blob: &mut [u8]) -> Result<CUgraphNode> {
        let api = self.dev.api();
        let mut size = 0;
        let mut extra = [std::ptr::null_mut(); 5];
        let params = Self::params(node, blob, &mut size, &mut extra);
        let mut handle = CUgraphNode::NULL;
        // SAFETY: the params and the blob they point to outlive the call,
        // which copies them.
        unsafe { (api.graph_add_kernel_node)(&mut handle, self.graph, deps.as_ptr(), deps.len(), &params) }
            .check("cuGraphAddKernelNode")?;
        Ok(handle)
    }

    fn add_event_node(&self, deps: &[CUgraphNode], event: &CudaEvent) -> Result<CUgraphNode> {
        let api = self.dev.api();
        let mut handle = CUgraphNode::NULL;
        // SAFETY: a live graph, dependency handles of it, and a live event.
        unsafe { (api.graph_add_event_record_node)(&mut handle, self.graph, deps.as_ptr(), deps.len(), event.raw()) }
            .check("cuGraphAddEventRecordNode")?;
        Ok(handle)
    }

    /// Re-point the kernel nodes whose bindings changed.
    fn patch(&mut self, nodes: &[Node], blobs: &mut [Vec<u8>], buffers: &[u64], vals: &[i64]) -> Result<()> {
        if self.packed.0 == buffers && self.packed.1 == vals {
            return Ok(());
        }
        let api = self.dev.enter()?;
        let (mut buffer_offset, mut var_offset) = (0, 0);
        for (index, node) in nodes.iter().enumerate() {
            let (buffer_range, var_range) =
                (buffer_offset..buffer_offset + node.layout.globals, var_offset..var_offset + node.layout.vars);
            buffer_offset = buffer_range.end;
            var_offset = var_range.end;
            let changed = self.packed.0[buffer_range.clone()] != buffers[buffer_range.clone()]
                || self.packed.1[var_range.clone()] != vals[var_range.clone()];
            if !changed {
                continue;
            }
            node.layout.pack(&mut blobs[index], &buffers[buffer_range], &vals[var_range])?;
            let mut size = 0;
            let mut extra = [std::ptr::null_mut(); 5];
            let params = Self::params(node, &mut blobs[index], &mut size, &mut extra);
            // SAFETY: as `add_kernel_node`; only future launches see the update.
            unsafe { (api.graph_exec_kernel_node_set_params)(self.exec, self.kernels[index], &params) }
                .check("cuGraphExecKernelNodeSetParams")?;
        }
        self.packed = (buffers.to_vec(), vals.to_vec());
        Ok(())
    }

    /// Fresh event pairs for the next profiled launch, so handles already
    /// handed out keep their stamps.
    fn rearm_stamps(&mut self) -> Result<Vec<(Arc<CudaEvent>, Arc<CudaEvent>)>> {
        let api = self.dev.enter()?;
        let mut events = Vec::with_capacity(self.stamps.len());
        for (start_node, end_node) in &self.stamps {
            let start = Arc::new(CudaEvent::new(Arc::clone(&self.dev), true)?);
            let end = Arc::new(CudaEvent::new(Arc::clone(&self.dev), true)?);
            // SAFETY: record nodes of this exec and live events.
            unsafe {
                (api.graph_exec_event_record_node_set_event)(self.exec, *start_node, start.raw())
                    .check("cuGraphExecEventRecordNodeSetEvent")?;
                (api.graph_exec_event_record_node_set_event)(self.exec, *end_node, end.raw())
                    .check("cuGraphExecEventRecordNodeSetEvent")?;
            }
            events.push((start, end));
        }
        self.events = events.clone();
        Ok(events)
    }
}

impl Graph for CudaGraph {
    fn replay(&self, buffers: &[u64], vals: &[i64]) -> Result<()> {
        let (buffers, vals) = self.bindings(buffers, vals)?;
        let mut state = self.state.lock();
        let State { blobs, dag, chain, .. } = &mut *state;
        let exec = if alias_signature(buffers) == self.aliasing {
            dag
        } else {
            if chain.is_none() {
                tracing::debug!("CUDA graph replay with re-aliased buffers; using the capture-order chain");
                *chain = Some(Exec::build(&self.dev, &self.nodes, blobs, Topology::Chain, false, buffers, vals)?);
            }
            chain.as_mut().expect("built above")
        };
        exec.patch(&self.nodes, blobs, buffers, vals)?;
        state.last = Some(self.launch(exec)?);
        Ok(())
    }

    fn completion_token(&self) -> Option<Arc<dyn CompletionToken>> {
        Some(Arc::new(self.state.lock().last.clone()?))
    }

    fn replay_profiled(&self, buffers: &[u64], vals: &[i64]) -> Result<Option<Vec<Arc<dyn DispatchTimestamps>>>> {
        let (buffers, vals) = self.bindings(buffers, vals)?;
        let mut state = self.state.lock();
        let State { blobs, profiled, .. } = &mut *state;
        if profiled.is_none() {
            *profiled = Some(Exec::build(&self.dev, &self.nodes, blobs, Topology::Chain, true, buffers, vals)?);
        }
        let exec = profiled.as_mut().expect("built above");
        exec.patch(&self.nodes, blobs, buffers, vals)?;
        let events = exec.rearm_stamps()?;
        let done = self.launch(exec)?;
        state.last = Some(done.clone());
        // Waited here, so nothing is left for the executor to publish.
        done.wait(0)?;
        done.published();
        Ok(Some(
            events
                .into_iter()
                .map(|(start, end)| {
                    Arc::new(CudaDispatchTimestamps::new(Arc::clone(&self.dev), start, end))
                        as Arc<dyn DispatchTimestamps>
                })
                .collect(),
        ))
    }
}
