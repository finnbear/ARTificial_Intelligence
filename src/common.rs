use image;
use image::imageops;
use image::Pixel;
use imageproc::drawing;
use imageproc::rect::Rect;
use num::cast::ToPrimitive;
use num::Bounded;
use rand;
use rand::Rng;
use std::time::Instant;

fn calculate_difference<Img: image::GenericImageView>(image1: &Img, image2: &Img) -> f32 {
    assert_eq!(image1.dimensions(), image2.dimensions());

    let mut diff: u64 = 0;

    for y in 0..image1.height() {
        for x in 0..image1.width() {
            let pixel1 = image1.get_pixel(x, y);
            let channels1 = pixel1.channels();
            let pixel2 = image2.get_pixel(x, y);
            let channels2 = pixel2.channels();
            for c in 0..channels1.len() - 1 {
                diff +=
                    (channels1[c].to_i16().unwrap() - channels2[c].to_i16().unwrap()).abs() as u64;
            }
        }
    }

    diff as f32
        / (image1.width()
            * image2.height()
            * 4
            * <<Img::Pixel as image::Pixel>::Subpixel as Bounded>::max_value()
                .to_u32()
                .unwrap()) as f32
}

pub fn impressionize(source: &image::RgbaImage) -> image::RgbaImage {
    let mut canvas = image::RgbaImage::new(source.width(), source.height());

    let mut rng = rand::thread_rng();
    let now = Instant::now();
    loop {
        let diff = calculate_difference(source, &canvas);
        //let alpha = calculate_opacity(&canvas);
        //println!("diff: {}, opacity: {}", diff, alpha);

        if diff < 0.005 || now.elapsed().as_secs() > 20 {
            break;
        }

        let mut stroke_size: i32 = rng.gen_range(4..8);

        if diff < 0.05 {
            stroke_size = stroke_size.min(2);
        }

        let mut tmp = image::RgbaImage::new(stroke_size as u32 * 2, stroke_size as u32 * 2);

        let x = (rng.gen_range(0..source.width()) as i32).clamp(
            tmp.width() as i32 / 2,
            canvas.width() as i32 - tmp.width() as i32 / 2,
        );
        let y = (rng.gen_range(0..source.height()) as i32).clamp(
            tmp.height() as i32 / 2,
            canvas.height() as i32 - tmp.height() as i32 / 2,
        );

        for dx in -stroke_size..=stroke_size {
            for dy in -stroke_size..=stroke_size {
                if rng.gen_bool(0.9) {
                    continue;
                }
                let dot_size = rng.gen_range(2..4).min(stroke_size);
                let mut color = *source.get_pixel(x as u32, y as u32);
                color.0[3] = 50;

                if rng.gen_bool(0.5) {
                    let rect = Rect::at(
                        tmp.width() as i32 / 2 + dx - dot_size / 2,
                        tmp.height() as i32 / 2 + dy - dot_size / 2,
                    )
                    .of_size(dot_size as u32, dot_size as u32);
                    drawing::draw_filled_rect_mut(&mut tmp, rect, color);
                } else {
                    let center = (tmp.width() as i32 / 2, tmp.height() as i32 / 2);
                    drawing::draw_filled_circle_mut(&mut tmp, center, dot_size, color);
                }

                imageops::overlay(
                    &mut canvas,
                    &tmp,
                    (x - tmp.height() as i32 / 2) as u32,
                    (y - tmp.height() as i32 / 2) as u32,
                )
            }
        }
    }

    for (_, _, pixel) in canvas.enumerate_pixels_mut() {
        pixel.0[3] = 255;
    }
    canvas
}
