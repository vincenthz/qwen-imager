use std::{
    collections::{BTreeSet, HashMap},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::{CancellationToken, DownloadProgress};
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{VarBuilder, var_builder::SimpleBackend};
use hf_hub::{
    Cache, Repo, RepoType,
    api::{Progress, sync::ApiBuilder},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// A published Qwen Image 2.1 checkpoint. All use the Diffusers layout and key
/// names; the MLX packs store most large linear layers affine-quantized.
/// Turbo is a distilled BF16 denoiser that samples a fixed 8-step schedule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Checkpoint {
    /// The original BF16 weights, about 32 GB.
    #[default]
    #[serde(rename = "bf16")]
    Original,
    /// MLX 4-bit denoiser blocks and text encoder, about 11 GB.
    #[serde(rename = "mlx-4bit")]
    Mlx4Bit,
    /// MLX 8-bit denoiser blocks and text encoder, about 18 GB.
    #[serde(rename = "mlx-8bit")]
    Mlx8Bit,
    /// Qwen-Image-2.1-Turbo BF16 weights, about 32 GB.
    #[serde(rename = "turbo")]
    Turbo,
}

impl Checkpoint {
    pub const ALL: [Self; 4] = [Self::Original, Self::Mlx4Bit, Self::Mlx8Bit, Self::Turbo];

    pub fn repo(self) -> &'static str {
        match self {
            Self::Original => "Qwen/Qwen-Image-2.1",
            Self::Mlx4Bit => "ddalcu/Qwen-Image-2.1-MLX-Serve-4bit",
            Self::Mlx8Bit => "ddalcu/Qwen-Image-2.1-MLX-Serve-8bit",
            Self::Turbo => "Qwen/Qwen-Image-2.1-Turbo",
        }
    }

    /// Pinned commit. Architecture and weights are validated together.
    pub fn revision(self) -> &'static str {
        match self {
            // Matches the checkpoint used by the original Python CLI.
            Self::Original => "790c92633540aa0cb11d9abf19eb46d861714758",
            Self::Mlx4Bit => "1cdbb51e8f9269ea9f81d76b7190547f9eb3c512",
            Self::Mlx8Bit => "dc21b8d3441eef8f50ee5915dd5ea74679bcf338",
            Self::Turbo => "d65dbc9a7e8f6b5479e33dee6030eaab2a906509",
        }
    }

    /// Short name, as used by the CLI's `--model` and the desktop settings file.
    pub fn name(self) -> &'static str {
        match self {
            Self::Original => "bf16",
            Self::Mlx4Bit => "mlx-4bit",
            Self::Mlx8Bit => "mlx-8bit",
            Self::Turbo => "turbo",
        }
    }

    /// Approximate download size in GB.
    pub fn download_gb(self) -> u32 {
        match self {
            Self::Original | Self::Turbo => 32,
            Self::Mlx4Bit => 11,
            Self::Mlx8Bit => 18,
        }
    }

    /// Sampling sigmas saved with the checkpoint (`sample_sigmas` in its
    /// `model_index.json`), before the terminal zero. They replace the
    /// step-count schedule, so requested steps are ignored.
    pub fn sample_sigmas(self) -> Option<&'static [f64]> {
        match self {
            Self::Turbo => Some(&[
                1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
            ]),
            _ => None,
        }
    }

    /// Denoising steps actually sampled for a request asking for `requested`.
    pub fn steps(self, requested: usize) -> usize {
        self.sample_sigmas().map_or(requested, <[f64]>::len)
    }

    /// The MLX packs publish no shard index; their shards are fixed per revision.
    /// Turbo's text encoder is one unindexed file.
    fn shards(self, component: &str) -> Option<&'static [&'static str]> {
        match (self, component) {
            (Self::Original, _) => None,
            (Self::Turbo, "text_encoder") => Some(&["model.safetensors"]),
            (Self::Turbo, _) => None,
            (_, "text_encoder") => Some(&[
                "model-00001-of-00004.safetensors",
                "model-00002-of-00004.safetensors",
                "model-00003-of-00004.safetensors",
                "model-00004-of-00004.safetensors",
            ]),
            (_, "transformer") => Some(&[
                "diffusion_pytorch_model-00001-of-00002.safetensors",
                "diffusion_pytorch_model-00002-of-00002.safetensors",
            ]),
            _ => None,
        }
    }
}

impl std::fmt::Display for Checkpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.repo())
    }
}

/// Memoize actual device tensors, not just the file mappings. Builder clones
/// share this cache; request-local layers can be dropped without reloading weights.
pub fn cached_builder(source: VarBuilder<'static>) -> VarBuilder<'static> {
    let dtype = source.dtype();
    let device = source.device().clone();
    VarBuilder::from_backend(
        Box::new(CachedWeights {
            source,
            tensors: Mutex::new(HashMap::new()),
        }),
        dtype,
        device,
    )
}

struct CachedWeights {
    source: VarBuilder<'static>,
    tensors: Mutex<HashMap<String, Tensor>>,
}

impl SimpleBackend for CachedWeights {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.get_unchecked(name, dtype, device)?;
        if tensor.shape() != &shape {
            candle_core::bail!(
                "cached weight {name}: expected {shape:?}, got {:?}",
                tensor.shape()
            );
        }
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        // A cache belongs to one model's device/dtype for its entire lifetime.
        if !stored_dtype(dtype, self.source.dtype()) || !device.same_device(self.source.device()) {
            candle_core::bail!("cached weights require their original device and dtype");
        }
        let mut tensors = self
            .tensors
            .lock()
            .map_err(|_| candle_core::Error::Msg("weight cache lock poisoned".into()))?;
        if let Some(tensor) = tensors.get(name) {
            return Ok(tensor.clone());
        }
        let tensor = self.source.get_unchecked_dtype(name, dtype)?;
        tensors.insert(name.to_owned(), tensor.clone());
        Ok(tensor)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.source.contains_tensor(name)
    }
}

/// Weights load in the builder's float dtype, except MLX-packed quantized
/// values, which are read as unconverted `u32` words.
pub(crate) fn stored_dtype(requested: DType, builder: DType) -> bool {
    requested == builder || requested == DType::U32
}

#[derive(Clone)]
pub struct Weights {
    local: Option<PathBuf>,
    offline: bool,
    checkpoint: Checkpoint,
    shared: Option<Arc<crate::shared::SharedWeights>>,
}

impl Weights {
    pub fn new(local: Option<PathBuf>, offline: bool, checkpoint: Checkpoint) -> Self {
        Self {
            local,
            offline,
            checkpoint,
            shared: None,
        }
    }

    pub fn shared(local: Option<PathBuf>, offline: bool, checkpoint: Checkpoint) -> Self {
        Self {
            shared: Some(Arc::default()),
            ..Self::new(local, offline, checkpoint)
        }
    }

    pub fn file(&self, name: &str) -> Result<PathBuf> {
        self.file_with_progress(name, &mut |_| {})
    }

    fn file_with_progress(
        &self,
        name: &str,
        on_progress: &mut dyn FnMut(DownloadProgress),
    ) -> Result<PathBuf> {
        if let Some(root) = &self.local {
            let path = root.join(name);
            ensure!(
                path.is_file(),
                "missing checkpoint file: {}",
                path.display()
            );
            return Ok(path);
        }
        let (model, revision) = (self.checkpoint.repo(), self.checkpoint.revision());
        let repo = Repo::with_revision(model.into(), RepoType::Model, revision.into());
        let cache = std::env::var_os("HF_HUB_CACHE")
            .map(|p| Cache::new(p.into()))
            .unwrap_or_else(Cache::from_env);
        // Python's hub client does not create refs/<commit> for a pinned revision.
        let snapshot = cache
            .path()
            .join(repo.folder_name())
            .join("snapshots")
            .join(revision)
            .join(name);
        if snapshot.is_file() {
            return Ok(snapshot);
        }
        if let Some(path) = cache.repo(repo.clone()).get(name) {
            return Ok(path);
        }
        ensure!(
            !self.offline,
            "{name} is not cached; allow downloads or supply a local model directory"
        );
        on_progress(DownloadProgress {
            file: name.into(),
            downloaded: 0,
            total: None,
        });
        ApiBuilder::from_env()
            .with_cache_dir(cache.path().clone())
            .with_progress(false)
            .build()?
            .repo(repo)
            .download_with_progress(name, DownloadReporter::new(on_progress))
            .with_context(|| format!("downloading {model}/{name}"))
    }

    pub fn prepare(
        &self,
        cancellation: &CancellationToken,
        on_progress: &mut dyn FnMut(DownloadProgress),
    ) -> Result<()> {
        let mut fetch = |name: &str| -> Result<PathBuf> {
            cancellation.check()?;
            let path = self.file_with_progress(name, on_progress)?;
            cancellation.check()?;
            Ok(path)
        };
        for name in [
            "transformer/config.json",
            "processor/tokenizer.json",
            "vae/config.json",
        ] {
            fetch(name)?;
        }
        for component in ["text_encoder", "transformer", "vae"] {
            for name in self.shard_files(component, &mut fetch)? {
                fetch(&name)?;
            }
        }
        Ok(())
    }

    /// Checkpoint-relative paths of a component's safetensors files, read from
    /// the shard index when the checkpoint has one. Local directories without an
    /// index use every `.safetensors` file in the component's directory.
    fn shard_files(
        &self,
        component: &str,
        fetch: &mut dyn FnMut(&str) -> Result<PathBuf>,
    ) -> Result<Vec<String>> {
        if component == "vae" {
            return Ok(vec!["vae/diffusion_pytorch_model.safetensors".into()]);
        }
        let base = if component == "text_encoder" {
            "model"
        } else {
            "diffusion_pytorch_model"
        };
        let index = format!("{component}/{base}.safetensors.index.json");
        let names = match (&self.local, self.checkpoint.shards(component)) {
            (Some(root), _) if !root.join(&index).is_file() => {
                let mut names = Vec::new();
                let dir = root.join(component);
                for entry in
                    std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))?
                {
                    let name = entry?.file_name().to_string_lossy().into_owned();
                    if name.ends_with(".safetensors") {
                        names.push(name);
                    }
                }
                ensure!(
                    !names.is_empty(),
                    "no {component} safetensors files in {}",
                    dir.display()
                );
                names.sort();
                names
            }
            (None, Some(shards)) => shards.iter().map(|&s| s.to_owned()).collect(),
            _ => shard_names(&read_json(&fetch(&index)?)?)?
                .into_iter()
                .map(str::to_owned)
                .collect(),
        };
        Ok(names
            .into_iter()
            .map(|name| format!("{component}/{name}"))
            .collect())
    }

    pub fn checkpoint(&self) -> Checkpoint {
        self.checkpoint
    }

    pub fn config<T: DeserializeOwned>(&self, name: &str) -> Result<T> {
        read_json(&self.file(name)?)
    }

    pub fn builder(
        &self,
        component: &str,
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'static>> {
        if let Some(shared) = &self.shared {
            return shared.builder(self, component, dtype, device);
        }
        self.uncached_builder(component, dtype, device)
    }

    pub(crate) fn uncached_builder(
        &self,
        component: &str,
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'static>> {
        let files = self
            .shard_files(component, &mut |name| self.file(name))?
            .iter()
            .map(|name| self.file(name))
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: checkpoint files are read-only for the lifetime of this process. Do not modify
        // files in --model-dir while inference is running. The builder owns its mmap handles.
        unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, device) }
            .with_context(|| format!("loading {component} weights"))
    }
}

fn shard_names(index: &serde_json::Value) -> Result<BTreeSet<&str>> {
    let map = index["weight_map"]
        .as_object()
        .context("missing checkpoint weight_map")?;
    ensure!(!map.is_empty(), "checkpoint weight_map is empty");
    map.values()
        .map(|v| v.as_str().context("invalid weight filename"))
        .collect()
}

struct DownloadReporter<'a> {
    callback: &'a mut dyn FnMut(DownloadProgress),
    progress: DownloadProgress,
    last_emit: Instant,
}

impl<'a> DownloadReporter<'a> {
    fn new(callback: &'a mut dyn FnMut(DownloadProgress)) -> Self {
        Self {
            callback,
            progress: DownloadProgress {
                file: String::new(),
                downloaded: 0,
                total: None,
            },
            last_emit: Instant::now(),
        }
    }
    fn emit(&mut self) {
        (self.callback)(self.progress.clone());
        self.last_emit = Instant::now();
    }
}

impl Progress for DownloadReporter<'_> {
    fn init(&mut self, size: usize, filename: &str) {
        self.progress = DownloadProgress {
            file: filename.into(),
            downloaded: 0,
            total: Some(size as u64),
        };
        self.emit();
    }
    fn update(&mut self, size: usize) {
        self.progress.downloaded += size as u64;
        if self.last_emit.elapsed() >= Duration::from_millis(100) {
            self.emit();
        }
    }
    fn finish(&mut self) {
        self.emit();
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    serde_json::from_reader(std::fs::File::open(path)?)
        .with_context(|| format!("reading {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_weights_reuse_device_tensors_and_validate_requests() -> Result<()> {
        // The source creates a fresh tensor on every read. Identity and read
        // counts prove that reconstructed layers reuse the same device storage.
        let loads = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        struct Source(std::sync::Arc<std::sync::atomic::AtomicUsize>);
        impl SimpleBackend for Source {
            fn get(
                &self,
                _: Shape,
                name: &str,
                _: candle_nn::Init,
                dtype: DType,
                dev: &Device,
            ) -> candle_core::Result<Tensor> {
                self.get_unchecked(name, dtype, dev)
            }
            fn get_unchecked(
                &self,
                name: &str,
                dtype: DType,
                dev: &Device,
            ) -> candle_core::Result<Tensor> {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if name == "missing" {
                    candle_core::bail!("missing tensor");
                }
                Tensor::zeros(2, dtype, dev)
            }
            fn contains_tensor(&self, name: &str) -> bool {
                name != "missing"
            }
        }
        let cached = cached_builder(VarBuilder::from_backend(
            Box::new(Source(loads.clone())),
            DType::F32,
            Device::Cpu,
        ));
        let first = cached.pp("decoder").get(2, "weight")?;
        let id = first.id();
        drop(first);
        assert_eq!(cached.clone().pp("decoder").get(2, "weight")?.id(), id);
        assert!(cached.pp("decoder").get(3, "weight").is_err());
        assert!(
            cached
                .to_dtype(DType::BF16)
                .pp("decoder")
                .get(2, "weight")
                .is_err()
        );
        assert_eq!(loads.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert!(cached.contains_tensor("decoder.weight"));
        assert!(!cached.contains_tensor("missing"));
        assert!(cached.get(2, "missing").is_err());
        assert!(cached.get(2, "missing").is_err());
        assert_eq!(loads.load(std::sync::atomic::Ordering::Relaxed), 3);
        assert_ne!(cached.pp("encoder").get(2, "weight")?.id(), id);
        assert_eq!(loads.load(std::sync::atomic::Ordering::Relaxed), 4);
        // MLX-packed weights are read as u32, without conversion to the float dtype.
        let packed = cached.get_unchecked_dtype("packed.weight", DType::U32)?;
        assert_eq!(packed.dtype(), DType::U32);
        let id = packed.id();
        assert_eq!(
            cached
                .get_unchecked_dtype("packed.weight", DType::U32)?
                .id(),
            id
        );
        assert!(cached.get_unchecked_dtype("other", DType::F16).is_err());
        Ok(())
    }

    // Tests own their directories; never alter the user's model cache.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("img-gen-weights-{}", rand::random::<u64>()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn prepare_checks_all_shards_and_uses_local_files_without_downloads() -> Result<()> {
        let directory = TempDir::new();
        let weights = Weights::new(Some(directory.0.clone()), true, Checkpoint::Original);
        for component in ["processor", "text_encoder", "transformer", "vae"] {
            std::fs::create_dir(directory.0.join(component))?;
        }
        for name in [
            "processor/tokenizer.json",
            "transformer/config.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ] {
            std::fs::write(directory.0.join(name), b"fixture")?;
        }
        for (component, base) in [
            ("text_encoder", "model"),
            ("transformer", "diffusion_pytorch_model"),
        ] {
            std::fs::write(directory.0.join(format!("{component}/{base}.safetensors.index.json")),
                br#"{"weight_map":{"a":"one.safetensors","b":"two.safetensors","c":"one.safetensors"}}"#)?;
            for name in ["one.safetensors", "two.safetensors"] {
                std::fs::write(directory.0.join(component).join(name), b"fixture")?;
            }
        }
        let cancellation = CancellationToken::default();
        weights.prepare(&cancellation, &mut |_| {
            panic!("local file triggered download")
        })?;
        std::fs::remove_file(directory.0.join("transformer/two.safetensors"))?;
        let error = weights
            .prepare(&cancellation, &mut |_| panic!("offline download"))
            .unwrap_err();
        assert!(error.to_string().contains("two.safetensors"));
        cancellation.cancel();
        assert!(
            weights
                .prepare(&cancellation, &mut |_| {})
                .unwrap_err()
                .is::<crate::Cancelled>()
        );
        Ok(())
    }

    #[test]
    fn mlx_checkpoints_use_fixed_shards_and_local_directories_without_an_index() -> Result<()> {
        assert_eq!(Checkpoint::default(), Checkpoint::Original);
        for checkpoint in Checkpoint::ALL {
            assert_eq!(checkpoint.revision().len(), 40);
            let json = serde_json::to_string(&checkpoint)?;
            assert_eq!(json, format!("\"{}\"", checkpoint.name()));
        }
        let mut no_index = |name: &str| -> Result<PathBuf> { panic!("fetched {name}") };
        let remote = Weights::new(None, true, Checkpoint::Mlx4Bit);
        assert_eq!(
            remote.shard_files("transformer", &mut no_index)?,
            [
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
                "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            ]
        );
        assert_eq!(remote.shard_files("text_encoder", &mut no_index)?.len(), 4);
        let turbo = Weights::new(None, true, Checkpoint::Turbo);
        assert_eq!(
            turbo.shard_files("text_encoder", &mut no_index)?,
            ["text_encoder/model.safetensors"]
        );

        let directory = TempDir::new();
        let local = Weights::new(Some(directory.0.clone()), true, Checkpoint::Mlx8Bit);
        for name in [
            "processor/tokenizer.json",
            "transformer/config.json",
            "transformer/b.safetensors",
            "transformer/a.safetensors",
            "transformer/notes.txt",
            "text_encoder/model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
        ] {
            let path = directory.0.join(name);
            std::fs::create_dir_all(path.parent().unwrap())?;
            std::fs::write(path, b"fixture")?;
        }
        assert_eq!(
            local.shard_files("transformer", &mut no_index)?,
            ["transformer/a.safetensors", "transformer/b.safetensors"]
        );
        local.prepare(&CancellationToken::default(), &mut |_| {
            panic!("local file triggered download")
        })?;
        std::fs::remove_file(directory.0.join("text_encoder/model.safetensors"))?;
        assert!(
            local
                .prepare(&CancellationToken::default(), &mut |_| {})
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn download_progress_counts_resumed_bytes_and_resets_on_retry() {
        let mut events = Vec::new();
        let mut callback = |event| events.push(event);
        let mut progress = DownloadReporter::new(&mut callback);
        progress.init(100, "model.safetensors");
        progress.update(40); // hf-hub reports bytes already in its .part file.
        progress.update(20);
        progress.init(100, "model.safetensors"); // new attempt resets its counter.
        progress.update(60);
        progress.update(40);
        progress.finish();
        let last = events.last().unwrap();
        assert_eq!(last.downloaded, 100);
        assert_eq!(last.total, Some(100));
        assert_eq!(last.file, "model.safetensors");
    }

    #[test]
    #[ignore = "requires permission to bind a loopback HTTP fixture server"]
    fn missing_file_download_reports_actual_bytes() -> Result<()> {
        use std::{
            io::{Read, Write},
            net::TcpListener,
            thread,
        };
        let directory = TempDir::new();
        let server = TcpListener::bind("127.0.0.1:0")?;
        let endpoint = format!("http://{}", server.local_addr()?);
        let data = vec![42_u8; 65536];
        let expected = data.clone();
        let worker = thread::spawn(move || {
            for metadata in [true, false] {
                let (mut stream, _) = server.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .unwrap();
                let mut request = Vec::new();
                loop {
                    let mut byte = [0];
                    stream.read_exact(&mut byte).unwrap();
                    request.push(byte[0]);
                    if request.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                let body = if metadata { &data[..1] } else { &data[..] };
                write!(stream, "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes 0-{}/{}\r\nX-Repo-Commit: {}\r\nETag: fixture\r\nConnection: close\r\n\r\n",
                    body.len(), body.len() - 1, data.len(), Checkpoint::Original.revision()).unwrap();
                stream.write_all(body).unwrap();
            }
        });
        let mut events = Vec::new();
        let api = ApiBuilder::new()
            .with_endpoint(endpoint)
            .with_cache_dir(directory.0.clone())
            .with_progress(false)
            .build()?;
        let checkpoint = Checkpoint::Original;
        let repo = Repo::with_revision(
            checkpoint.repo().into(),
            RepoType::Model,
            checkpoint.revision().into(),
        );
        let path = api.repo(repo).download_with_progress(
            "fixture.safetensors",
            DownloadReporter::new(&mut |event| events.push(event)),
        )?;
        worker.join().unwrap();
        assert_eq!(std::fs::read(path)?, expected);
        assert_eq!(events.first().unwrap().downloaded, 0);
        let last = events.last().unwrap();
        assert_eq!(last.downloaded, expected.len() as u64);
        assert_eq!(last.total, Some(expected.len() as u64));
        Ok(())
    }
}
