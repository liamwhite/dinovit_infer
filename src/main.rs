use clap::Parser;
use image::{DynamicImage, ImageReader, RgbImage};
use image::{ImageBuffer, Pixel};
use smallvec::SmallVec;
use std::error::Error;
use tch::jit::{CModule, IValue};
use tch::Tensor;

/// Each DINOv2 patch is 14x14
const PATCH_DIM: i64 = 14;

/// - CLS for dinov2
#[allow(dead_code)]
const DINOV2_EMBEDDINGS_OFFSET: i64 = 1;

/// - CLS + 4*REG for dinov2-with-registers
const DINOV2_WITH_REG_EMBEDDINGS_OFFSET: i64 = 1 + 4;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    pytorch_jit_model: String,

    #[arg(long)]
    image: String,
}

struct ModelResult {
    pub patches: (i64, i64),
    #[allow(dead_code)]
    pub image: Tensor,
    pub pooler_output: Tensor,
    pub last_hidden_state: Tensor,
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

    // Get channels.
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
    //
    // TODO: unnecessary device pin
    let ones = Tensor::ones([3, h, w], (tch::Kind::Float, tch::Device::Cpu));
    let mask = alpha.where_self(&color.gt_tensor(&alpha).any(), &ones);
    let color = color.multiply(&mask);

    // Pure transparency is rescaled to be 8 steps blacker than black.
    const ALPHA_LEVEL: f64 = 8.0 / 255.0;
    const COLOR_LEVEL: f64 = 1.0 - ALPHA_LEVEL;

    Ok(color
        .multiply_scalar(COLOR_LEVEL)
        .f_add(&alpha.multiply_scalar(ALPHA_LEVEL))?)
}

fn resize_tensor(image: Tensor, size: (i64, i64)) -> Tensor {
    image.upsample_bicubic2d([size.0, size.1], true, None, None)
}

fn load_image(path: &str) -> Result<Tensor, Box<dyn Error>> {
    let image = ImageReader::open(path)?.with_guessed_format()?.decode()?;
    let image = strip_transparency(image)?;

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

fn infer(image: &Tensor, model: &CModule) -> Result<(Tensor, Tensor), Box<dyn Error>> {
    let output = model.forward_is(&[IValue::Tensor(image.shallow_clone())])?;
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
    let scaled_norm = pooler_output.norm().pow_tensor_scalar(-1);
    pooler_output.multiply(&scaled_norm)
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
    embeddings_offset: i64,
    size: (i64, i64),
) -> Result<Tensor, Box<dyn Error>> {
    // discard CLS token, we just want patch embeddings
    let a = last_hidden_state
        .slice(1, embeddings_offset, None, 1)
        .squeeze();
    let pc = into_principal_components(&a, 3)?;

    // normalize
    let max = pc.max_dim(0, false).0;
    let min = pc.min_dim(0, false).0;
    let range = max.f_sub(&min)?;
    let pc = pc.f_sub(&min)?.f_div(&range)?;

    // arrange into input shape and put channels first
    Ok(pc.reshape([size.0, size.1, 3]).permute([2, 0, 1]))
}

fn resize_image_by_patch_count(image: Tensor, patches: (i64, i64)) -> Tensor {
    let height = patches.0 * PATCH_DIM;
    let width = patches.1 * PATCH_DIM;

    resize_tensor(image.unsqueeze(0), (height, width))
}

fn get_model_result(
    image_path: &str,
    image_scale: i64,
    model: &CModule,
) -> Result<ModelResult, Box<dyn Error>> {
    // Get image and and dimensions for calculation.
    let image = load_image(image_path)?;

    // Features are unstable across different global scales, and
    // somewhat stable across dimensional scales.
    //
    // Use 18 (252x252) instead of 16 (224x224) to produce a more detailed
    // result and attention map at almost exactly the same computational cost.
    //
    // It is possible for highly non-square models to have meaningful feature extraction,
    // but in practice it makes no difference identifying scales which keep the aspect
    // ratio, and does not produce feature vectors which are similar enough to identify
    // crops.
    let patches = (18 * image_scale, 18 * image_scale);

    // Scale image into appropriate shape.
    let image = resize_image_by_patch_count(image, patches);

    // The pooler output is the [CLS] token generated by the model.
    // It contains high-quality, robust features from the input image.
    let (last_hidden_state, pooler_output) = infer(&image, model)?;

    Ok(ModelResult {
        patches,
        image,
        pooler_output: pooler_output.squeeze(),
        last_hidden_state,
    })
}

fn get_features_and_attention(
    image_path: &str,
    image_scale: i64,
    embeddings_offset: i64,
    model: &CModule,
) -> Result<(Tensor, Tensor), Box<dyn Error>> {
    let result = get_model_result(image_path, image_scale, model)?;
    let features = scaled_result(&result.pooler_output);
    let attention =
        visualize_attention(&result.last_hidden_state, embeddings_offset, result.patches)?;

    Ok((features, attention))
}

#[allow(dead_code)]
fn print_time<R, F: FnOnce() -> R>(label: &str, f: F) -> R {
    let begin = std::time::Instant::now();
    let r = f();
    let end = std::time::Instant::now();
    let duration = (end - begin).as_secs_f64();
    eprintln!("{}: {}", label, duration);
    r
}

#[allow(dead_code)]
fn print_single_dimensional_tensor(data: &Tensor) -> Result<(), Box<dyn Error>> {
    use itertools::Itertools;

    let size = data.size1()?;
    let output: Vec<f64> = (0..size).map(|i| data.double_value(&[i])).collect();

    println!("[{}]", output.iter().join(","));

    Ok(())
}

fn console_evaluate(args: &Args) -> Result<(), Box<dyn Error>> {
    let model = CModule::load(&args.pytorch_jit_model)?;
    let (features, attention) =
        get_features_and_attention(&args.image, 1, DINOV2_WITH_REG_EMBEDDINGS_OFFSET, &model)?;

    print_single_dimensional_tensor(&features)?;
    save_image("/tmp/attention.png", &attention)?;

    Ok(())
}

fn main() {
    let args = Args::parse();
    tch::set_num_threads(4);
    tch::no_grad(|| console_evaluate(&args).unwrap());
}
