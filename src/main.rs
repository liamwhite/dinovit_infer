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

fn resize_tensor(image: Tensor, size: (i64, i64)) -> Tensor {
    image.upsample_bicubic2d([size.0, size.1], true, None, None)
}

fn load_image(path: &str, size: (i64, i64)) -> Result<Tensor, Box<dyn Error>> {
    let image = ImageReader::open(path)?.with_guessed_format()?.decode()?;
    let image = strip_transparency(image)?.unsqueeze(0);
    let image = resize_tensor(image, size);

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

fn infer(image: Tensor, model: &CModule) -> Result<(Tensor, Tensor), Box<dyn Error>> {
    let output = model.forward_is(&[IValue::Tensor(image)])?;
    let mut results = match output {
        IValue::Tuple(elements) if elements.len() == 2 => elements,
        _ => return Err("expected (last_hidden_state, pooler_output)".into()),
    };

    let mut results = results.drain(..);

    match (results.next(), results.next()) {
        (Some(IValue::Tensor(last_hidden_state)), Some(IValue::Tensor(pooler_output))) => {
            Ok((last_hidden_state, pooler_output))
        }
        _ => Err("expected 2-tuple of tensors".into()),
    }
}

fn scaled_result(pooler_output: &Tensor) -> Tensor {
    let scaled_norm = pooler_output
        .norm()
        .pow_tensor_scalar(-1)
        .multiply_scalar(128.0);

    pooler_output
        .multiply(&scaled_norm)
        .clamp(-128.0, 127.0)
        .to_kind(tch::Kind::Int8)
}

#[allow(dead_code)]
fn into_principal_components(data: &Tensor, q: i64) -> Result<Tensor, Box<dyn Error>> {
    let mean = data.mean_dim(0, false, None);
    let centered_data = data.f_sub(&mean)?;
    let (_u, _s, v) = centered_data.svd(true, true);

    Ok(centered_data.matmul(&v.slice(1, 0, q, 1)))
}

#[allow(dead_code)]
fn visualize_attention(
    last_hidden_state: &Tensor,
    size: (i64, i64),
) -> Result<Tensor, Box<dyn Error>> {
    // discard CLS token, we just want patch embeddings
    let a = last_hidden_state.slice(1, 1, None, 1).squeeze();
    let pc = into_principal_components(&a, 3)?;

    // normalize
    let max = pc.max_dim(0, false).0;
    let min = pc.min_dim(0, false).0;
    let range = max.f_sub(&min)?;
    let pc = pc.f_sub(&min)?.f_div(&range)?;

    // arrange into input shape and put channels first
    Ok(pc.reshape([size.0, size.1, 3]).permute([2, 0, 1]))
}

fn console_evaluate(args: &Args) -> Result<(), Box<dyn Error>> {
    let image = load_image(&args.image, (224, 224))?;
    let model = CModule::load(&args.pytorch_jit_model)?;

    // TODO: evaluate what subset of features give good "similarity" results
    // Do we just want to evaluate the CLS token, or all of the patch tokens as well?
    let (last_hidden_state, pooler_output) = infer(image, &model)?;
    println!("{}", scaled_result(&pooler_output.squeeze()).to_string(80)?);
    save_image(
        "/tmp/attention.png",
        &visualize_attention(&last_hidden_state, (16, 16))?,
    )?;

    Ok(())
}

fn main() {
    let args = Args::parse();
    tch::set_num_threads(4);
    tch::no_grad(|| console_evaluate(&args).unwrap());
}
