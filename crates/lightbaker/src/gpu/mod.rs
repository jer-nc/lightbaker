//! The GPU side: one device, the scene's buffers, and the three passes of trace.wgsl run in batches
//! (each batch submitted and waited on, so no dispatch runs long enough for the driver to reset it).

use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::bvh::Bvh;
use crate::raster::Texel;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Params {
    pub sun_dir: [f32; 4],
    pub sun_irr: [f32; 4],
    pub zenith: [f32; 4],
    pub horizon: [f32; 4],
    pub ground: [f32; 4],
    pub fallback: [f32; 4],
    pub counts: [u32; 4],
    pub reach: [f32; 4],
    pub atlas: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    /// per texel: soft sun, hard sun bits, occlusion, back faces
    Direct,
    /// per texel: irradiance rgb, back faces
    Gather,
    /// per probe, four values: L1 coefficients (rgb) and the sun, back faces
    Volume,
}

impl Pass {
    fn entry(self) -> &'static str {
        match self {
            Pass::Direct => "direct",
            Pass::Gather => "gather",
            Pass::Volume => "volume",
        }
    }
    /// results an item writes
    fn width(self) -> usize {
        if self == Pass::Volume { 4 } else { 1 }
    }
    /// items a dispatch: the costlier the item, the fewer
    fn batch(self) -> usize {
        match self {
            Pass::Direct => 65536,
            Pass::Gather => 16384,
            Pass::Volume => 8192,
        }
    }
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 3],
    pub adapter: String,
}

/// The scene as the passes read it.
pub struct SceneBuffers {
    nodes: wgpu::Buffer,
    tris: wgpu::Buffer,
    materials: wgpu::Buffer,
    charts: wgpu::Buffer,
}

impl Gpu {
    pub fn new() -> Result<Gpu> {
        pollster::block_on(Self::open())
    }

    async fn open() -> Result<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
            .await
            .context("no GPU adapter")?;
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor { label: Some("lightbaker"), required_limits: adapter.limits(), ..Default::default() })
            .await
            .context("no GPU device")?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("trace"), source: wgpu::ShaderSource::Wgsl(include_str!("trace.wgsl").into()) });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trace"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                storage(4, true),
                storage(5, false),
                storage(6, true),
                storage(7, true),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("trace"), bind_group_layouts: &[Some(&layout)], ..Default::default() });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = [pipeline("direct"), pipeline("gather"), pipeline("volume")];
        Ok(Gpu { device, queue, layout, pipelines, adapter: format!("{} ({:?})", info.name, info.backend) })
    }

    pub fn scene(&self, bvh: &Bvh, materials: &[[f32; 4]], charts: &[[[f32; 4]; 2]]) -> SceneBuffers {
        SceneBuffers {
            nodes: self.storage("nodes", bytemuck::cast_slice(&bvh.nodes)),
            tris: self.storage("tris", bytemuck::cast_slice(&bvh.tris)),
            materials: self.storage("materials", bytemuck::cast_slice(materials)),
            charts: self.storage("charts", bytemuck::cast_slice(if charts.is_empty() { &[[[0f32; 4]; 2]] } else { charts })),
        }
    }

    /// Runs `pass` over `texels` (probes for the volume: their position, their index as the normal's w),
    /// reading `light` (per atlas texel); returns each item's results.
    pub fn run(&self, pass: Pass, scene: &SceneBuffers, params: Params, texels: &[Texel], light: &[[f32; 4]]) -> Result<Vec<[f32; 4]>> {
        let items = texels.len();
        let mut out = Vec::with_capacity(items * pass.width());
        if items == 0 {
            return Ok(out);
        }
        let texels = self.storage("texels", bytemuck::cast_slice(texels));
        let light = self.storage("light", bytemuck::cast_slice(if light.is_empty() { &[[0f32; 4]] } else { light }));
        let pipeline = &self.pipelines[pass as usize];
        let mut first = 0;
        while first < items {
            let n = pass.batch().min(items - first);
            let mut p = params;
            p.counts[3] = n as u32;
            p.atlas[2] = first as u32;
            let uniform = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("params"), contents: bytemuck::bytes_of(&p), usage: wgpu::BufferUsages::UNIFORM });
            let size = (n * pass.width() * 16) as u64;
            let result = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("out"), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let read = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("read"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            // the batch's slice of the texels (or probes)
            let (offset, tsize) = ((first * 64) as u64, std::num::NonZeroU64::new((n * 64) as u64));
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("trace"),
                layout: &self.layout,
                entries: &[
                    entry(0, &scene.nodes, 0, None),
                    entry(1, &scene.tris, 0, None),
                    entry(2, &scene.materials, 0, None),
                    entry(3, &uniform, 0, None),
                    entry(4, &texels, offset, tsize),
                    entry(5, &result, 0, None),
                    entry(6, &scene.charts, 0, None),
                    entry(7, &light, 0, None),
                ],
            });
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("trace") });
            {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(pass.entry()), timestamp_writes: None });
                cp.set_pipeline(pipeline);
                cp.set_bind_group(0, &group, &[]);
                cp.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
            }
            enc.copy_buffer_to_buffer(&result, 0, &read, 0, size);
            self.queue.submit([enc.finish()]);
            let slice = read.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow!("GPU: {e:?}"))?;
            rx.recv()?.map_err(|e| anyhow!("GPU read back: {e:?}"))?;
            out.extend_from_slice(bytemuck::cast_slice(&slice.get_mapped_range()?));
            read.unmap();
            first += n;
        }
        Ok(out)
    }

    fn storage(&self, label: &str, contents: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage: wgpu::BufferUsages::STORAGE })
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer, offset: u64, size: Option<std::num::NonZeroU64>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset, size }) }
}
