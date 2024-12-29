
use clap::Parser;
use std::{error::Error, time::Instant};
use tch::vision::imagenet;
use tch::jit::{CModule, IValue};

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

fn infer(args: &Args) -> Result<(), Box<dyn Error>> {
    tch::set_num_threads(4);

    let image = imagenet::load_image_and_resize(&args.image, 224, 224)?;
    let model = CModule::load(&args.pytorch_jit_model)?;

    let begin = Instant::now();
    let output = model.forward_is(&[IValue::Tensor(image.unsqueeze(0))])?;
    let end = Instant::now();
    let duration = (end - begin).as_secs_f64();

    println!("Evaluated in {duration} seconds {output:?}");

    Ok(())
}
