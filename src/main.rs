
use clap::Parser;
use std::{error::Error, time::Instant};
use tch::{nn::Module, vision::imagenet};

mod dinov2;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,

    #[arg(long)]
    image: String,
}

fn main() {
    let args = Args::parse();
    infer(&args).unwrap();
}

fn infer(args: &Args) -> Result<(), Box<dyn Error>> {
    let image = imagenet::load_image_and_resize(&args.image, dinov2::IMG_SIZE, dinov2::IMG_SIZE)?;
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let net = Box::new(dinov2::vit_base(vs.root(), None));
    vs.load(&args.model)?;

    let begin = Instant::now();
    let output = net.forward(&image.unsqueeze(0));
    let end = Instant::now();
    let duration = (end - begin).as_secs_f64();

    println!("Evaluated in {duration} seconds {output:?}");

    Ok(())
}
