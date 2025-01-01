use clap::{Parser, Subcommand};
use std::error::Error;
use std::fs::OpenOptions;
use std::io::stdout;

mod batch_inference;
#[allow(dead_code)]
mod dinov2;
#[allow(dead_code)]
mod io;

#[derive(Parser)]
struct Args {
    /// Path to the model to use.
    model_path: String,

    /// Scale to use.
    #[arg(default_value_t = 1)]
    image_scale: i64,

    /// Task subtype.
    #[command(subcommand)]
    task: Task,
}

#[derive(Clone, Subcommand)]
enum Task {
    /// Run a single inference job, printing the output and attention map.
    InferSingle {
        /// Image to load.
        image_path: String,

        /// Path for generated attention map.
        output_attention_path: String,
    },

    /// Runs two inference jobs, printing the cosine distance between them.
    CosineSimilarity {
        /// First image to load.
        image1_path: String,

        /// Second image to load.
        image2_path: String,
    },

    /// Runs full batch inference job.
    BatchInference {
        /// Base location of files.
        base_path: String,

        /// Path to file containing input JSON lines.
        input_json: String,

        /// Path to file to create containing output features.
        output_json: String,

        /// Number of threads to use.
        num_threads: usize,
    },
}

fn infer_single(
    model_path: &str,
    image_path: &str,
    output_attention_path: &str,
    image_scale: i64,
) -> Result<(), Box<dyn Error>> {
    let (device, model) = io::device_and_model(model_path)?;

    let image_file = OpenOptions::new().read(true).open(image_path)?;
    let mut attention_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_attention_path)?;

    let result = dinov2::get_model_result(image_file, image_scale, &model, device)?;
    io::write_tensor(&mut stdout().lock(), &result.features)?;
    println!();

    let attention = dinov2::visualize_attention(
        &result.last_hidden_state,
        dinov2::DINOV2_WITH_REG_EMBEDDINGS_OFFSET,
        result.patches,
    )?;

    io::save_image(&mut attention_file, &attention)
}

fn cosine_similarity(
    model_path: &str,
    image1_path: &str,
    image2_path: &str,
    image_scale: i64,
) -> Result<(), Box<dyn Error>> {
    let (device, model) = io::device_and_model(model_path)?;

    let image1_file = OpenOptions::new().read(true).open(image1_path)?;
    let image2_file = OpenOptions::new().read(true).open(image2_path)?;

    let features1 = dinov2::get_model_result(image1_file, image_scale, &model, device)?.features;
    let features2 = dinov2::get_model_result(image2_file, image_scale, &model, device)?.features;

    let cosine_sim = features1
        .dot(&features2)
        .divide(&features1.norm().multiply(&features2.norm()));
    println!("{}", cosine_sim.double_value(&[]));

    Ok(())
}

#[rustfmt::skip]
fn console_execute() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    match args.task {
        Task::InferSingle { image_path, output_attention_path } => infer_single(
            &args.model_path,
            &image_path,
            &output_attention_path,
            args.image_scale,
        ),
        Task::CosineSimilarity { image1_path, image2_path } => cosine_similarity(
            &args.model_path,
            &image1_path,
            &image2_path,
            args.image_scale,
        ),
        Task::BatchInference { base_path, input_json, output_json, num_threads } => batch_inference::run(
            &args.model_path,
            &base_path,
            &input_json,
            &output_json,
            args.image_scale,
            num_threads,
        ),
    }
}

fn main() {
    tch::set_num_threads(4);
    tch::no_grad(|| console_execute().unwrap());
}
