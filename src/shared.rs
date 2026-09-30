//! Immutable model buffers shared by independent Metal inference queues.
use crate::weights::Weights;
use anyhow::Result;
use candle_core::{DType, Device, Shape, Storage, Tensor, metal_backend::MetalStorage};
use candle_nn::{VarBuilder, var_builder::SimpleBackend};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Weak},
};

#[derive(Default)]
pub(crate) struct SharedWeights {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    resident: HashMap<String, Arc<SharedTensors>>,
    // Share encoders while they are in use, then release them. Keeping the
    // entire text encoder resident would add ~16 GiB to idle memory use.
    encoder: Weak<SharedTensors>,
}

impl SharedWeights {
    pub fn builder(
        &self,
        weights: &Weights,
        component: &str,
        dtype: DType,
        device: &Device,
    ) -> Result<VarBuilder<'static>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("shared model lock poisoned"))?;
        let existing = if component == "text_encoder" {
            state.encoder.upgrade()
        } else {
            state.resident.get(component).cloned()
        };
        let tensors = match existing {
            Some(tensors) => tensors,
            None => {
                // Each component has a loading queue, serialized by its tensor
                // cache lock. Workspaces never enqueue work on these queues.
                let loader = Device::new_metal(0)?;
                let source = weights.uncached_builder(component, dtype, &loader)?;
                let tensors = Arc::new(SharedTensors {
                    source,
                    tensors: Mutex::new(HashMap::new()),
                });
                if component == "text_encoder" {
                    state.encoder = Arc::downgrade(&tensors);
                } else {
                    state.resident.insert(component.to_owned(), tensors.clone());
                }
                tensors
            }
        };
        anyhow::ensure!(
            dtype == tensors.source.dtype(),
            "shared weights require their original dtype"
        );
        Ok(tensors.builder(device.clone()))
    }
}

struct SharedTensors {
    source: VarBuilder<'static>,
    tensors: Mutex<HashMap<String, Tensor>>,
}

impl SharedTensors {
    fn builder(self: &Arc<Self>, device: Device) -> VarBuilder<'static> {
        VarBuilder::from_backend(
            Box::new(SharedBackend {
                shared: self.clone(),
            }),
            self.source.dtype(),
            device,
        )
    }

    fn get(&self, name: &str, dtype: DType, device: &Device) -> candle_core::Result<Tensor> {
        if dtype != self.source.dtype() {
            candle_core::bail!("shared weights require their original dtype");
        }
        let tensor = {
            let mut tensors = self
                .tensors
                .lock()
                .map_err(|_| candle_core::Error::Msg("shared tensor cache poisoned".into()))?;
            if let Some(tensor) = tensors.get(name) {
                tensor.clone()
            } else {
                let tensor = immutable_buffer(self.source.get_unchecked(name)?)?;
                tensors.insert(name.to_owned(), tensor.clone());
                tensor
            }
        };
        on_queue(&tensor, device)
    }
}

struct SharedBackend {
    shared: Arc<SharedTensors>,
}
impl SimpleBackend for SharedBackend {
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
                "shared weight {name}: expected {shape:?}, got {:?}",
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
        self.shared.get(name, dtype, device)
    }
    fn contains_tensor(&self, name: &str) -> bool {
        self.shared.source.contains_tensor(name)
    }
}

// Ordinary Candle buffers belong to a per-queue recycling allocator. A buffer
// shared across queues must NOT go back into that allocator when one alias is
// dropped. Private buffers are excluded from Candle's allocator; Metal's own
// retain count then keeps the allocation alive across all workspace aliases.
fn immutable_buffer(tensor: Tensor) -> candle_core::Result<Tensor> {
    let Device::Metal(device) = tensor.device() else {
        return Ok(tensor);
    };
    let (storage, layout) = tensor.storage_and_layout();
    if !layout.is_contiguous() || layout.start_offset() != 0 {
        candle_core::bail!("shared checkpoint tensors must be contiguous with zero offset");
    }
    let Storage::Metal(storage) = &*storage else {
        unreachable!()
    };
    let buffer =
        device.new_private_buffer(tensor.elem_count(), tensor.dtype(), "shared model weight")?;
    {
        let encoder = device.blit_command_encoder()?;
        encoder.copy_from_buffer(
            storage.buffer(),
            0,
            &buffer,
            0,
            tensor.elem_count() * tensor.dtype().size_in_bytes(),
        );
        // Blit encoders require explicit completion (unlike compute encoders).
        encoder.end_encoding();
    }
    // Publish only after loading, dtype conversion and the private copy finish.
    device.wait_until_completed()?;
    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            buffer,
            device.clone(),
            tensor.elem_count(),
            tensor.dtype(),
        )),
        tensor.shape().clone(),
    )))
}

fn on_queue(tensor: &Tensor, target: &Device) -> candle_core::Result<Tensor> {
    if tensor.device().same_device(target) {
        return Ok(tensor.clone());
    }
    match (tensor.device(), target) {
        (Device::Metal(source), Device::Metal(target))
            if source.registry_id() == target.registry_id() =>
        {
            let (storage, _) = tensor.storage_and_layout();
            let Storage::Metal(storage) = &*storage else {
                unreachable!()
            };
            // Cloning the Metal handle aliases the allocation; no pixel/weight
            // data is copied. Only queue/allocator metadata belongs to the tab.
            Ok(Tensor::from((
                Storage::Metal(MetalStorage::new(
                    Arc::new(storage.buffer().clone()),
                    target.clone(),
                    tensor.elem_count(),
                    tensor.dtype(),
                )),
                tensor.shape().clone(),
            )))
        }
        _ => candle_core::bail!("shared weights require queues on the same physical device"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn concurrent_builders_load_each_weight_once() -> Result<()> {
        struct Source(Arc<AtomicUsize>);
        impl SimpleBackend for Source {
            fn get(
                &self,
                _: Shape,
                name: &str,
                _: candle_nn::Init,
                dtype: DType,
                device: &Device,
            ) -> candle_core::Result<Tensor> {
                self.get_unchecked(name, dtype, device)
            }
            fn get_unchecked(&self, _: &str, _: DType, _: &Device) -> candle_core::Result<Tensor> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Tensor::new(&[1f32, 2.], &Device::Cpu)
            }
            fn contains_tensor(&self, _: &str) -> bool {
                true
            }
        }
        let loads = Arc::new(AtomicUsize::new(0));
        let shared = Arc::new(SharedTensors {
            source: VarBuilder::from_backend(
                Box::new(Source(loads.clone())),
                DType::F32,
                Device::Cpu,
            ),
            tensors: Mutex::new(HashMap::new()),
        });
        let workers: Vec<_> = (0..4)
            .map(|_| {
                let builder = shared.builder(Device::Cpu);
                std::thread::spawn(move || builder.get(2, "weight"))
            })
            .collect();
        let tensors: Vec<_> = workers
            .into_iter()
            .map(|t| t.join().unwrap().unwrap())
            .collect();
        assert!(tensors.iter().all(|t| t.id() == tensors[0].id()));
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert!(shared.builder(Device::Cpu).get(3, "weight").is_err());
        assert!(
            shared
                .builder(Device::Cpu)
                .to_dtype(DType::BF16)
                .get(2, "weight")
                .is_err()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires Metal GPU access"]
    fn metal_weight_aliases_share_storage_across_independent_queues() -> Result<()> {
        let loader = Device::new_metal(0)?;
        let first = Device::new_metal(0)?;
        let second = Device::new_metal(0)?;
        let source = immutable_buffer(Tensor::new(&[1f32, 2., 3., 4.], &loader)?)?;
        let a = on_queue(&source, &first)?;
        let b = on_queue(&source, &second)?;
        assert!(!a.device().same_device(b.device()));
        {
            let (a_storage, _) = a.storage_and_layout();
            let (b_storage, _) = b.storage_and_layout();
            let (Storage::Metal(a), Storage::Metal(b)) = (&*a_storage, &*b_storage) else {
                unreachable!()
            };
            assert_eq!(a.buffer(), b.buffer());
        }
        // Aliases must survive both the source's release and allocator reuse.
        drop(source);
        for _ in 0..20 {
            let _ = Tensor::zeros(4, DType::F32, &loader)?;
        }
        loader.synchronize()?;
        let one = std::thread::spawn(move || (&a * 2.)?.to_vec1::<f32>());
        let two = std::thread::spawn(move || (&b * 3.)?.to_vec1::<f32>());
        assert_eq!(one.join().unwrap()?, [2., 4., 6., 8.]);
        assert_eq!(two.join().unwrap()?, [3., 6., 9., 12.]);
        Ok(())
    }
}
