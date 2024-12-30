use clap::Parser;
use image::{DynamicImage, ImageReader, RgbImage};
use image::{ImageBuffer, Pixel};
use smallvec::SmallVec;
use std::error::Error;
use tch::jit::{CModule, IValue};
use tch::Tensor;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    pytorch_jit_model: String,

    #[arg(long)]
    image: String,
}

fn main() {
    let args = Args::parse();
    infer(&args).unwrap();
}

fn into_tensor<P: Pixel<Subpixel = f32>>(image: ImageBuffer<P, Vec<f32>>) -> Tensor {
    let w: i64 = image.width().into();
    let h: i64 = image.height().into();
    let c: i64 = P::CHANNEL_COUNT.into();

    // Extra scope to ensure we eagerly drop the original image buffer
    let pixels = {
        let pixels: Vec<f32> = image.pixels().flat_map(|p| p.channels()).copied().collect();

        Tensor::from_slice(&pixels)
    };

    pixels.reshape([h, w, c]).permute([2, 0, 1])
}

fn strip_transparency(image: DynamicImage) -> Result<Tensor, Box<dyn Error>> {
    let w: i64 = image.width().into();
    let h: i64 = image.height().into();

    match image {
        DynamicImage::ImageRgb8(..)
        | DynamicImage::ImageLuma8(..)
        | DynamicImage::ImageLuma16(..)
        | DynamicImage::ImageRgb16(..)
        | DynamicImage::ImageRgb32F(..) => {
            return Ok(into_tensor(image.into_rgb32f()));
        }
        _ => {}
    };

    // Get channels
    let (alpha, color) = {
        let pixels = into_tensor(image.into_rgba32f());
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
    let threshold_count = color
        .gt_tensor(&alpha)
        .sum(tch::Kind::Int64)
        .int64_value(&[]);

    let color = if threshold_count > 0 {
        color.multiply(&alpha)
    } else {
        color
    };

    // Pure transparency is rescaled to be 8 steps "blacker than black"
    const ALPHA_LEVEL: f64 = 8.0 / 255.0;
    const COLOR_LEVEL: f64 = 1.0 - ALPHA_LEVEL;

    Ok(color
        .multiply_scalar(COLOR_LEVEL)
        .f_add(&alpha.multiply_scalar(ALPHA_LEVEL))?)
}

fn resize_tensor(image: Tensor, width: i64, height: i64) -> Tensor {
    image.upsample_bicubic2d([width, height], true, None, None)
}

fn load_image(path: &str) -> Result<Tensor, Box<dyn Error>> {
    let image = ImageReader::open(path)?.with_guessed_format()?.decode()?;
    let image = strip_transparency(image)?.unsqueeze(0);
    let image = resize_tensor(image, 224, 224);

    Ok(image)
}

#[allow(dead_code)]
fn save_image(path: &str, image: &Tensor) -> Result<(), Box<dyn Error>> {
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

    Ok(output_image.save(path)?)
}

fn infer(args: &Args) -> Result<(), Box<dyn Error>> {
    tch::set_num_threads(4);

    let image = load_image(&args.image)?;
    let model = CModule::load(&args.pytorch_jit_model)?;
    let output = model.forward_is(&[IValue::Tensor(image)])?;

    let results = match output {
        IValue::Tuple(elements) if elements.len() == 2 => elements,
        _ => return Err("expected (last_hidden_state, pooler_output)".into()),
    };

    let IValue::Tensor(last_hidden_state) = &results[0] else {
        return Err("expected first tuple element to be a tensor".into());
    };

    let infer_result = last_hidden_state.mean_dim(1, false, None).squeeze();
    let scaled_norm = infer_result
        .norm()
        .pow_tensor_scalar(-1)
        .multiply_scalar(128);
    let infer_result = infer_result
        .multiply(&scaled_norm)
        .to_dtype(tch::Kind::Int8, false, true);

    let Ok(768) = infer_result.size1() else {
        return Err("last_hidden_state size is not 768".into());
    };

    for i in 0..768 {
        print!("{} ", infer_result.int64_value(&[i]));
    }
    println!("");

    Ok(())
}
