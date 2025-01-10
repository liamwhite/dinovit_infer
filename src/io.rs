use image::{DynamicImage, ImageBuffer, ImageFormat, ImageReader, Pixel, RgbImage};
use smallvec::SmallVec;
use std::time::Instant;
use std::{error::Error, io::BufReader};
use tch::{CModule, Device, Tensor};

pub fn device_and_model(model_path: &str) -> Result<(Device, CModule), Box<dyn Error>> {
    let device = Device::cuda_if_available();
    let model = CModule::load_on_device(model_path, device)?;

    Ok((device, model))
}

fn into_tensor<P: Pixel<Subpixel = f32>>(
    image: ImageBuffer<P, Vec<f32>>,
    device: Device,
) -> Tensor {
    let w: i64 = image.width().into();
    let h: i64 = image.height().into();
    let c: i64 = P::CHANNEL_COUNT.into();

    // Extra scope to ensure we eagerly drop the original image buffer
    let pixels = {
        let pixels: Vec<f32> = image.pixels().flat_map(|p| p.channels()).copied().collect();

        Tensor::from_slice(&pixels)
    };

    pixels.to(device).reshape([h, w, c]).permute([2, 0, 1])
}

fn strip_transparency(image: DynamicImage, device: Device) -> Result<Tensor, Box<dyn Error>> {
    let w: i64 = image.width().into();
    let h: i64 = image.height().into();

    match image {
        DynamicImage::ImageRgb8(..)
        | DynamicImage::ImageLuma8(..)
        | DynamicImage::ImageLuma16(..)
        | DynamicImage::ImageRgb16(..)
        | DynamicImage::ImageRgb32F(..) => {
            return Ok(into_tensor(image.into_rgb32f(), device));
        }
        _ => {}
    };

    // Get channels.
    let (alpha, color) = {
        let pixels = into_tensor(image.into_rgba32f(), device);
        let alpha = pixels.slice(0, 3, 4, 1).broadcast_to([3, h, w]);
        let color = pixels.slice(0, 0, 3, 1);

        (alpha, color)
    };

    // Detect whether premultiplication should be applied by checking
    // for channels with values above the alpha level.
    //
    // Note that the only input format which we can get where this would
    // be relevant, PNG, explicitly says it does not carry premultiplied alpha,
    // but many tools will store premultiplied alpha anyway...
    let ones = Tensor::ones([3, h, w], (tch::Kind::Float, device));
    let mask = alpha.where_self(&color.gt_tensor(&alpha).any(), &ones);
    let color = color.multiply(&mask);

    // Pure transparency is rescaled to be 8 steps blacker than black.
    const ALPHA_LEVEL: f64 = 8.0 / 255.0;
    const COLOR_LEVEL: f64 = 1.0 - ALPHA_LEVEL;

    Ok(color
        .multiply_scalar(COLOR_LEVEL)
        .f_add(&alpha.multiply_scalar(ALPHA_LEVEL))?)
}

pub fn load_image<R>(image: R, device: Device) -> Result<Tensor, Box<dyn Error>>
where
    R: std::io::Read + std::io::Seek,
{
    let image = BufReader::new(image);
    let image = ImageReader::new(image).with_guessed_format()?.decode()?;
    let image = strip_transparency(image, device)?;

    Ok(image)
}

fn resize_tensor(image: Tensor, size: (i64, i64)) -> Tensor {
    image.upsample_bicubic2d([size.0, size.1], true, None, None)
}

pub fn resize_image_by_patch_count(image: Tensor, patches: (i64, i64), patch_dim: i64) -> Tensor {
    let height = patches.0 * patch_dim;
    let width = patches.1 * patch_dim;

    resize_tensor(image.unsqueeze(0), (height, width))
}

pub fn save_image<S>(sink: &mut S, image: &Tensor) -> Result<(), Box<dyn Error>>
where
    S: std::io::Write + std::io::Seek,
{
    let image = image
        .squeeze()
        .multiply_scalar(255.0)
        .clamp(0.0, 255.0)
        .to_kind(tch::Kind::Uint8);

    let (c, h, w) = image.size3()?;
    let mut output_image = RgbImage::new(w as u32, h as u32);

    for y in 0..h {
        for x in 0..w {
            let values: SmallVec<[u8; 4]> = (0..c)
                .map(|z| image.int64_value(&[z, y, x]) as u8)
                .collect();

            output_image.put_pixel(x as u32, y as u32, *Pixel::from_slice(&values));
        }
    }

    Ok(output_image.write_to(sink, ImageFormat::Png)?)
}

pub fn time<R, F>(f: F) -> (f64, R)
where
    F: FnOnce() -> R,
{
    let begin = Instant::now();
    let r = f();
    let end = Instant::now();

    ((end - begin).as_secs_f64(), r)
}

pub fn write_tensor<S>(sink: &mut S, data: &Tensor) -> Result<(), Box<dyn Error>>
where
    S: std::io::Write,
{
    let mut first = true;
    let iter = data.iter::<f64>()?;

    write!(sink, "[")?;

    for v in iter {
        if first {
            write!(sink, "{}", v)?;
            first = false;
        } else {
            write!(sink, ",{}", v)?;
        }
    }

    write!(sink, "]")?;

    Ok(())
}
