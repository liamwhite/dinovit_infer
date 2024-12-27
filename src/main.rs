use std::error::Error;

use candle_core::{Device, DType, Module};
use candle_nn::VarBuilder;
use candle_transformers::models::dinov2reg4;
use clap::Parser;

mod imagenet;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,

    #[arg(long)]
    image: String,
}

fn main() {
    let args = Args::parse();
    let device = Device::Cpu;

    infer(&args, device).unwrap();
}

fn infer(args: &Args, device: Device) -> Result<(), Box<dyn Error>> {
    let image = imagenet::load_image518(&args.image)?.to_device(&device)?;
    println!("loaded image {image:?}");

    let model_file = core::slice::from_ref(&args.model);
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(model_file, DType::F32, &device)? };
    let model = dinov2reg4::vit_base(vb)?;
    let logits = model.forward(&image.unsqueeze(0)?)?;
    println!("logits {logits:?}");

    Ok(())
}
