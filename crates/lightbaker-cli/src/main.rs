//! `lightbaker <scene.lbscene>… [--out dir] [--cache dir]`: bakes each scene into `<out>/<name>.lbake`,
//! reusing the tiles the cache already holds. One GPU device for all of them.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use lightbaker::{Gpu, Options, Scene, bake, output};

#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// scene files (.lbscene)
    #[arg(required = true)]
    scenes: Vec<PathBuf>,
    /// where the bakes are written (default: beside each scene)
    #[arg(long)]
    out: Option<PathBuf>,
    /// where baked tiles are kept between runs (default: none, everything baked)
    #[arg(long)]
    cache: Option<PathBuf>,
    /// say less
    #[arg(long, short)]
    quiet: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let gpu = Gpu::new()?;
    if !args.quiet {
        eprintln!("GPU: {}", gpu.adapter);
    }
    for path in &args.scenes {
        let started = Instant::now();
        let name = path.file_stem().context("scene file name")?.to_string_lossy().to_string();
        let scene = Scene::read(&std::fs::read(path).with_context(|| format!("reading {}", path.display()))?).with_context(|| format!("{}", path.display()))?;
        let quiet = args.quiet;
        let log = move |s: &str| {
            if !quiet {
                eprintln!("  {s}");
            }
        };
        if !args.quiet {
            eprintln!("{name}:");
        }
        let baked = bake(&scene, &gpu, &Options { cache: args.cache.as_deref(), log: &log })?;
        let dir = args.out.clone().unwrap_or_else(|| path.parent().map(PathBuf::from).unwrap_or_default());
        std::fs::create_dir_all(&dir)?;
        let file = dir.join(format!("{name}.lbake"));
        std::fs::write(&file, output::write(&scene, &baked)?)?;
        println!("{name}: {} of {} tiles baked, {:.1} s -> {}", baked.stats.tiles_baked, baked.stats.tiles, started.elapsed().as_secs_f64(), file.display());
    }
    Ok(())
}
