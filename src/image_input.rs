//! Reference-image loading shared by the desktop, CLI, and HTTP service.
//! Decoders are portable Rust; formats are detected from their contents.

use std::{io::Cursor, path::Path};

use image::{DynamicImage, ImageDecoder, ImageError, ImageFormat, ImageResult, Limits, RgbaImage};

/// Extensions accepted by the desktop import dialog.
pub const EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "heic", "heif"];

/// Load PNG, JPEG, WebP, or HEIC/HEIF into straight-alpha RGBA pixels.
pub fn open(path: impl AsRef<Path>) -> ImageResult<RgbaImage> {
    let bytes = std::fs::read(path)?;
    ImageInput::new(&bytes, Limits::default())?.decode()
}

/// An inspected input. Callers can reserve memory using `dimensions` before decoding.
pub struct ImageInput<'a>(Decoder<'a>);

enum Decoder<'a> {
    Standard(Box<dyn ImageDecoder + 'a>),
    Heic {
        bytes: &'a [u8],
        dimensions: (u32, u32),
        options: heic_rs::DecodeOptions,
    },
}

impl<'a> ImageInput<'a> {
    pub fn new(bytes: &'a [u8], mut limits: Limits) -> ImageResult<Self> {
        let decoder = match image::guess_format(bytes) {
            Ok(format @ (ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP)) => {
                let mut reader = image::ImageReader::with_format(Cursor::new(bytes), format);
                reader.limits(limits.clone());
                Decoder::Standard(Box::new(reader.into_decoder()?))
            }
            _ => {
                if heic_rs::ftyp::parse(bytes).is_err() {
                    return Err(unsupported());
                }
                let info = heic_rs::probe(bytes).map_err(heic_error)?;
                limits.check_dimensions(info.coded_width, info.coded_height)?;
                limits.check_dimensions(info.width, info.height)?;
                let options = heic_rs::DecodeOptions::default()
                    .with_layout(heic_rs::PixelLayout::Rgba8)
                    .with_strict(true)
                    .with_max_pixels(Some(
                        limits
                            .max_alloc
                            .map_or(heic_rs::DEFAULT_MAX_PIXELS, |bytes| bytes / 4)
                            .min(heic_rs::DEFAULT_MAX_PIXELS),
                    ));
                heic_rs::image::check_pixels(
                    info.coded_width,
                    info.coded_height,
                    options.pixel_limit(),
                )
                .map_err(heic_error)?;
                heic_rs::image::check_pixels(info.width, info.height, options.pixel_limit())
                    .map_err(heic_error)?;
                Decoder::Heic {
                    bytes,
                    dimensions: (info.width, info.height),
                    options,
                }
            }
        };
        let input = Self(decoder);
        let (width, height) = input.dimensions();
        if width == 0 || height == 0 {
            return Err(decode_error("image must not be empty"));
        }
        let rgba_bytes = u64::from(width)
            .saturating_mul(u64::from(height))
            .saturating_mul(4);
        let allocation = match &input.0 {
            Decoder::Standard(decoder) => decoder.total_bytes().max(rgba_bytes),
            Decoder::Heic { .. } => rgba_bytes,
        };
        limits.reserve(allocation)?;
        Ok(input)
    }

    /// Dimensions before JPEG EXIF orientation; rotation may swap width and height.
    /// HEIF container transforms are already reflected in these dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        match &self.0 {
            Decoder::Standard(decoder) => decoder.dimensions(),
            Decoder::Heic { dimensions, .. } => *dimensions,
        }
    }

    pub fn decode(self) -> ImageResult<RgbaImage> {
        match self.0 {
            Decoder::Standard(mut decoder) => {
                let orientation = decoder.orientation()?;
                let mut image = DynamicImage::from_decoder(decoder)?;
                image.apply_orientation(orientation);
                Ok(image.to_rgba8())
            }
            Decoder::Heic {
                bytes,
                dimensions,
                options,
            } => {
                let decoded = heic_rs::decode(bytes, &options).map_err(heic_error)?;
                if (decoded.width, decoded.height) != dimensions {
                    return Err(decode_error(
                        "HEIC decoded dimensions differ from its metadata",
                    ));
                }
                RgbaImage::from_raw(decoded.width, decoded.height, decoded.data)
                    .ok_or_else(|| decode_error("invalid HEIC pixel buffer"))
            }
        }
    }
}

fn decode_error(message: &str) -> ImageError {
    ImageError::Decoding(image::error::DecodingError::new(
        image::error::ImageFormatHint::Name("HEIC/HEIF".into()),
        message.to_owned(),
    ))
}

fn unsupported() -> ImageError {
    ImageError::Unsupported(image::error::UnsupportedError::from_format_and_kind(
        image::error::ImageFormatHint::Unknown,
        image::error::UnsupportedErrorKind::GenericFeature(
            "reference images must be PNG, JPEG, WebP, or HEIC/HEIF".into(),
        ),
    ))
}

fn heic_error(error: heic_rs::Error) -> ImageError {
    match error {
        heic_rs::Error::PixelLimit { .. } => ImageError::Limits(
            image::error::LimitError::from_kind(image::error::LimitErrorKind::InsufficientMemory),
        ),
        heic_rs::Error::Unsupported(message) => {
            ImageError::Unsupported(image::error::UnsupportedError::from_format_and_kind(
                image::error::ImageFormatHint::Name("HEIC/HEIF".into()),
                image::error::UnsupportedErrorKind::GenericFeature(message.into()),
            ))
        }
        error => decode_error(&error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const JPEG: &[u8] = include_bytes!("../tests/fixtures/oriented.jpg");
    const HEIC: &[u8] = include_bytes!("../tests/fixtures/oriented.heic");

    #[test]
    fn jpeg_and_heic_decode_with_correct_orientation_and_colors() {
        for bytes in [JPEG, HEIC] {
            let image = ImageInput::new(bytes, Limits::default())
                .unwrap()
                .decode()
                .unwrap();
            assert_eq!(image.dimensions(), (32, 64));
            for (x, y, expected) in [
                (8, 16, [20u8, 20, 240, 255]),
                (24, 16, [240, 20, 20, 255]),
                (8, 48, [240, 240, 20, 255]),
                (24, 48, [20, 240, 20, 255]),
            ] {
                let actual = image.get_pixel(x, y).0;
                assert!(
                    actual
                        .iter()
                        .zip(expected)
                        .all(|(&a, b)| a.abs_diff(b) <= 12),
                    "{actual:?} != {expected:?}"
                );
            }
        }
    }

    #[test]
    fn decodes_tiled_heic_at_full_resolution() {
        let bytes = include_bytes!("../tests/fixtures/tiled.heic");
        let info = heic_rs::probe(bytes).unwrap();
        assert!(info.is_grid);
        let image = ImageInput::new(bytes, Limits::default())
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!(image.dimensions(), (768, 1024));
        for (x, y, expected) in [
            (192, 256, [20u8, 20, 240, 255]),
            (576, 256, [240, 20, 20, 255]),
            (192, 768, [240, 240, 20, 255]),
            (576, 768, [20, 240, 20, 255]),
        ] {
            let actual = image.get_pixel(x, y).0;
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(&a, b)| a.abs_diff(b) <= 12),
                "{actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn loads_jpg_jpeg_heic_heif_and_uppercase_paths() {
        let directory =
            std::env::temp_dir().join(format!("image-forger-input-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        for (extension, bytes) in [
            ("jpg", JPEG),
            ("jpeg", JPEG),
            ("JPG", JPEG),
            ("JPEG", JPEG),
            ("heic", HEIC),
            ("HEIC", HEIC),
            ("heif", HEIC),
            ("HEIF", HEIC),
            ("dat", HEIC),
        ] {
            let path = directory.join(format!("reference.{extension}"));
            std::fs::write(&path, bytes).unwrap();
            assert_eq!(open(&path).unwrap().dimensions(), (32, 64));
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn png_and_webp_keep_straight_alpha() {
        let expected = RgbaImage::from_pixel(3, 2, image::Rgba([120, 80, 40, 60]));
        for format in [ImageFormat::Png, ImageFormat::WebP] {
            let mut bytes = Cursor::new(Vec::new());
            expected.write_to(&mut bytes, format).unwrap();
            let image = ImageInput::new(bytes.get_ref(), Limits::default())
                .unwrap()
                .decode()
                .unwrap();
            assert_eq!(image, expected);
        }
    }

    #[test]
    fn rejects_corrupt_unsupported_and_oversized_images() {
        assert!(matches!(
            ImageInput::new(b"GIF89a", Limits::default()),
            Err(ImageError::Unsupported(_))
        ));
        for bytes in [JPEG, HEIC] {
            assert!(
                ImageInput::new(&bytes[..bytes.len() / 2], Limits::default())
                    .and_then(ImageInput::decode)
                    .is_err()
            );
            let mut limits = Limits::default();
            limits.max_image_width = Some(16);
            limits.max_image_height = Some(16);
            assert!(matches!(
                ImageInput::new(bytes, limits),
                Err(ImageError::Limits(_))
            ));
            let mut limits = Limits::default();
            limits.max_alloc = Some(1);
            assert!(matches!(
                ImageInput::new(bytes, limits).and_then(ImageInput::decode),
                Err(ImageError::Limits(_))
            ));
        }
    }
}
