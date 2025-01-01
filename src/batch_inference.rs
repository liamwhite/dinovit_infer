use std::error::Error;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Lines, Write};
use std::sync::Arc;
use std::thread;

use chrono::{DateTime, Datelike, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tch::{CModule, Device};

use crate::{dinov2, io};

#[derive(Debug, Deserialize)]
struct Record {
    id: i64,
    format: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
struct Features {
    id: i64,
    features: Vec<f64>,
}

struct ProcessConfig {
    model: CModule,
    base_path: String,
    image_scale: i64,
    device: Device,
}

fn process_record(
    config: &ProcessConfig,
    record: &Record,
) -> Result<Option<Features>, Box<dyn Error>> {
    if record.format == "webm" {
        return Ok(None);
    }

    let id = record.id;
    let time = &record.created_at;
    let format = &record.format;
    let filename = format!(
        "{}/{}/{}/{}/{}.{}",
        config.base_path,
        time.year(),
        time.month(),
        time.day(),
        id,
        format
    );

    let image_file = OpenOptions::new().read(true).open(&filename)?;
    let result =
        dinov2::get_model_result(image_file, config.image_scale, &config.model, config.device)?;
    let features = result.features.iter::<f64>()?.collect();

    Ok(Some(Features { id, features }))
}

fn process_all_records<R, W>(
    config: Arc<ProcessConfig>,
    reader: Arc<Mutex<Lines<R>>>,
    writer: Arc<Mutex<W>>,
) where
    R: BufRead,
    W: Write,
{
    tch::no_grad(|| loop {
        let line = match reader.lock().next() {
            Some(line) => line.unwrap(),
            _ => return,
        };

        let record: Record = serde_json::from_str(&line).unwrap();

        match process_record(&config, &record) {
            Ok(None) => {}
            Ok(Some(features)) => {
                let mut writer = writer.lock();

                serde_json::to_writer(&mut *writer, &features).unwrap();
                writeln!(&mut writer).unwrap();
            }
            Err(e) => {
                eprintln!("\nError processing record {}: {}", record.id, e);
            }
        }
    });
}

pub fn run(
    model_path: &str,
    base_path: &str,
    input_json: &str,
    output_json: &str,
    image_scale: i64,
    num_threads: usize,
) -> Result<(), Box<dyn Error>> {
    tch::set_num_threads(1);

    let (device, model) = io::device_and_model(model_path)?;
    let config = Arc::new(ProcessConfig {
        model,
        base_path: base_path.to_owned(),
        image_scale,
        device,
    });

    let input_file = OpenOptions::new().read(true).open(input_json)?;
    let input_file = Arc::new(Mutex::new(BufReader::new(input_file).lines()));
    let output_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_json)?;
    let output_file = Arc::new(Mutex::new(output_file));

    let mut children = vec![];

    for _ in 0..num_threads {
        let config = config.clone();
        let input_file = input_file.clone();
        let output_file = output_file.clone();

        children.push(thread::spawn(move || {
            process_all_records(config, input_file, output_file)
        }))
    }

    for child in children {
        let _ = child.join();
    }

    Ok(())
}
