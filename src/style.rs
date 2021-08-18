use image;
mod common;

// This is for testing style-related algorithms.
fn main() {
    let source = image::io::Reader::open("output/1627796658803-0.61.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgba8();
    let canvas = common::impressionize(&source);

    canvas.save("style.png").unwrap();
}
