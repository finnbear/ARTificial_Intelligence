mod common;
use crossbeam::thread;
use image;
use image::imageops;
use imageproc;
use imageproc::drawing::Canvas;
use imageproc::rect::Rect;
use palette;
use palette::IntoColor;
use rand;
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::fs;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const REGISTERS: usize = 5;
const TRAIN_RESOLUTION: u32 = 128;
const TEST_RESOLUTION: u32 = 256;

#[derive(Clone, Serialize, Deserialize)]
struct Program {
    operations: Vec<Operation>,
}

impl Program {
    fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        for _ in 0..rng.gen_range(1..=3) {
            let range = 0..self.operations.len();
            match rng.gen_range(0..=3) {
                0 => self.operations[rng.gen_range(range)] = rng.gen(),
                1 => self.operations.insert(rng.gen_range(range), rng.gen()),
                2 => {
                    if self.operations.len() > 1 {
                        self.operations.remove(rng.gen_range(range));
                    }
                }
                3 => self
                    .operations
                    .swap(rng.gen_range(range.to_owned()), rng.gen_range(range)),
                //4 => self.operations[rng.gen_range(range)].mutate(),
                _ => unreachable!(),
            }
        }
    }

    fn combine(a: &Self, b: &Self) -> Self {
        let mut program = Self::new();

        let mut rng = rand::thread_rng();
        let mut which = false;
        let switch: f64 = rng.gen_range(0.01..0.2);
        let discard: bool = rng.gen();
        let mut iter_a = a.operations.iter();
        let mut iter_b = b.operations.iter();
        while iter_a.len() + iter_b.len() > 0 {
            if iter_b.len() == 0 || (which && iter_a.len() > 0) {
                program.operations.push(iter_a.next().unwrap().clone());
                if discard {
                    iter_b.next();
                }
            } else {
                program.operations.push(iter_b.next().unwrap().clone());
                if discard {
                    iter_a.next();
                }
            }
            if rng.gen_bool(switch) {
                which = !which;
            }
        }
        program
    }

    fn save(&self, path: String) {
        let program = serde_yaml::to_string(&self).unwrap();
        fs::write(path, program).unwrap();
    }

    fn load(&mut self, path: String) {
        let string = fs::read_to_string(path).unwrap();
        *self = serde_yaml::from_str(&*string).unwrap();
    }
}

impl Distribution<Program> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Program {
        let mut program = Program::new();

        for _ in 0..rng.gen_range(10..15) {
            program.operations.push(rng.gen());
        }

        program
    }
}

struct Process<'a> {
    context: Context,
    program: &'a Program,
}

impl<'a> Process<'a> {
    fn new(program: &'a Program) -> Self {
        Self {
            context: Context::new(1),
            program,
        }
    }

    // Returns (exists, ok)
    fn run_next(&mut self) -> Option<bool> {
        match self.program.operations.get(self.context.program_counter) {
            None => None,
            Some(operation) => {
                self.context.program_counter += 1;
                //println!("frame {} pc {}", self.context.frames.len(), self.context.frames.last().unwrap().program_counter);
                Some(operation.apply(&mut self.context))
            }
        }
    }

    fn run(&mut self, resolution: u32, max_instructions: i32, timeout: Duration) -> f32 {
        self.context = Context::new(resolution);

        let mut i = 0;
        let mut nok = 0;
        let now = Instant::now();
        while let Some(ok) = self.run_next() {
            i += 1;
            if !ok {
                nok += 1;
            }
            if i > max_instructions || self.context.frames.len() > 5 || now.elapsed() >= timeout {
                break;
            }
        }

        let image = &self.context.frames[0].image;
        let opacity = calculate_opacity(image);
        let entropy = calculate_entropy(image);
        let noise = calculate_noise(image);

        entropy
            * opacity
            * (1.0 - noise * 10.0).clamp(0.25, 1.0)
            * (1.0 - (now.elapsed().as_secs_f32() * 0.05).min(1.0))
            * (1.0 - (self.program.operations.len() as f32 * 0.005).min(1.0))
            * (1.0 - (nok as f32 / i as f32) * 0.25)
    }

    fn save(&self, path: String) {
        self.program.save(format!("{}.yml", path));
        self.context.save_image(&format!("{}.png", path));
    }
}

struct Frame {
    image: image::RgbaImage,
    registers: [u8; REGISTERS],
}

impl Frame {
    fn new(width: u32, height: u32) -> Self {
        Self {
            image: image::RgbaImage::new(width, height),
            registers: [0; REGISTERS],
        }
    }
}

struct Context {
    frames: Vec<Frame>,
    program_counter: usize,
    registers: [u8; REGISTERS],
}

impl Context {
    fn new(resolution: u32) -> Self {
        Self {
            frames: vec![Frame::new(resolution, resolution)],
            program_counter: 0,
            registers: [0; REGISTERS],
        }
    }

    fn image(&self) -> &image::RgbaImage {
        &self.frames.last().unwrap().image
    }

    fn image_mut(&mut self) -> &mut image::RgbaImage {
        &mut self.frames.last_mut().unwrap().image
    }

    fn scale(&self, x: u8) -> u32 {
        return x as u32 * self.image().dimensions().0 * self.image().dimensions().1 / (255 * 255);
    }

    fn scale_x(&self, x: u8) -> u32 {
        return x as u32 * self.image().dimensions().0 / 255;
    }

    fn scale_y(&self, y: u8) -> u32 {
        return y as u32 * self.image().dimensions().1 / 255;
    }

    fn save_image(&self, path: &String) {
        self.frames[0].image.save(path).unwrap();
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
enum Variable {
    Constant(u8),
    GlobalRegister(u8),
    FrameRegister(u8),
    Random { min: u8, max: u8 },
}

impl Variable {
    fn value(&self, context: &Context) -> u8 {
        match self {
            Variable::Constant(c) => *c,
            Variable::GlobalRegister(n) => context.registers[*n as usize],
            Variable::FrameRegister(n) => context.frames.last().unwrap().registers[*n as usize],
            Variable::Random { min, max } => rand::thread_rng().gen_range(*min..=*max),
        }
    }

    fn can_mut(&self) -> bool {
        match self {
            Variable::GlobalRegister(_) => true,
            Variable::FrameRegister(_) => true,
            _ => false,
        }
    }

    fn value_mut<'a>(&self, context: &'a mut Context) -> &'a mut u8 {
        match self {
            Variable::GlobalRegister(n) => &mut context.registers[*n as usize],
            Variable::FrameRegister(n) => {
                &mut context.frames.last_mut().unwrap().registers[*n as usize]
            }
            _ => panic!("not mutable"),
        }
    }

    fn bool_value(&self, context: &Context) -> bool {
        self.value(context) > u8::MAX / 2
    }

    fn f32_value(&self, context: &Context) -> f32 {
        self.value(context) as f32 * (1.0 / 255.0)
    }

    fn random_constant<R: Rng + ?Sized>(min: u8, max: u8, rng: &mut R) -> Variable {
        return Variable::Constant(rng.gen_range(min..=max));
    }

    fn random_register<R: Rng + ?Sized>(rng: &mut R) -> Variable {
        if rng.gen_bool(0.5) {
            Variable::GlobalRegister(rng.gen_range(0..REGISTERS as u8))
        } else {
            Variable::FrameRegister(rng.gen_range(0..REGISTERS as u8))
        }
    }

    fn random_random<R: Rng + ?Sized>(min: u8, max: u8, rng: &mut R) -> Variable {
        let m = rng.gen_range(min..max);
        return Variable::Random {
            min: m,
            max: rng.gen_range(m..=max),
        };
    }

    fn random<R: Rng + ?Sized>(rng: &mut R) -> Variable {
        match rng.gen_range(0..3) {
            0 => Variable::random_constant(u8::MIN, u8::MAX, rng),
            1 => Variable::random_register(rng),
            2 => Variable::random_random(u8::MIN, u8::MAX, rng),
            _ => unreachable!(),
        }
    }

    fn random_non_constant<R: Rng + ?Sized>(rng: &mut R) -> Variable {
        match rng.gen_range(0..2) {
            0 => Variable::random_register(rng),
            1 => Variable::random_random(u8::MIN, u8::MAX, rng),
            _ => unreachable!(),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
enum Operation {
    Push(Push),
    Pop(Pop),
    Jump(Jump),
    BranchNotEqualZero(BranchNotEqualZero),
    Set(Set),
    Add(Add),
    Subtract(Subtract),
    Multiply(Multiply),
    Size(Size),
    Scale(Scale),
    Blur(Blur),
    Pixel(Pixel),
    ReadPixel(ReadPixel),
    Line(Line),
    Circle(Circle),
    Ellipse(Ellipse),
    Rectangle(Rectangle),
}

impl Operation {
    fn apply(&self, context: &mut Context) -> bool {
        match self {
            Self::Push(push) => push.apply(context),
            Self::Pop(pop) => pop.apply(context),
            Self::Jump(jump) => jump.apply(context),
            Self::BranchNotEqualZero(brange_not_equal_zero) => brange_not_equal_zero.apply(context),
            Self::Set(set) => set.apply(context),
            Self::Add(add) => add.apply(context),
            Self::Subtract(subtract) => subtract.apply(context),
            Self::Multiply(multiply) => multiply.apply(context),
            Self::Size(size) => size.apply(context),
            Self::Scale(scale) => scale.apply(context),
            Self::Blur(blur) => blur.apply(context),
            Self::Pixel(pixel) => pixel.apply(context),
            Self::ReadPixel(read_pixel) => read_pixel.apply(context),
            Self::Line(line) => line.apply(context),
            Self::Circle(circle) => circle.apply(context),
            Self::Ellipse(ellipse) => ellipse.apply(context),
            Self::Rectangle(rectangle) => rectangle.apply(context),
        }
    }

    /*
    fn mutate(&mut self) {
        match self {
            Self::Push(push) => push.mutate(),
            Self::Pop(pop) => pop.mutate(),
            Self::Jump(jump) => jump.mutate(),
            Self::BranchNotEqualZero(brange_not_equal_zero) => brange_not_equal_zero.mutate(),
            Self::Set(set) => set.mutate(),
            Self::Add(add) => add.mutate(),
            Self::Subtract(subtract) => subtract.mutate(),
            Self::Multiply(multiply) => multiply.mutate(),
            Self::Size(size) => size.mutate(),
            Self::Scale(scale) => scale.mutate(),
            Self::Blur(blur) => blur.mutate(),
            Self::Pixel(pixel) => pixel.mutate(),
            Self::Line(line) => line.mutate(),
            Self::Circle(circle) => circle.mutate(),
            Self::Ellipse(ellipse) => ellipse.mutate(),
            Self::Rectangle(rectangle) => rectangle.mutate(),
        }
    }
     */
}

impl Distribution<Operation> for Standard {
    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> Operation {
        let mut rng = rand::thread_rng();
        match rng.gen_range(0..=13) {
            0 => Operation::Push(rng.gen()),
            1 => Operation::Pop(rng.gen()),
            2 => Operation::Jump(rng.gen()),
            3 => Operation::BranchNotEqualZero(rng.gen()),
            4 => Operation::Set(rng.gen()),
            5 => Operation::Add(rng.gen()),
            6 => Operation::Subtract(rng.gen()),
            7 => Operation::Multiply(rng.gen()),
            //8 => Operation::Size(rng.gen()),
            //9 => Operation::Scale(rng.gen()),
            //10 => Operation::Blur(rng.gen()),
            8 => Operation::Pixel(rng.gen()),
            9 => Operation::ReadPixel(rng.gen()),
            10 => Operation::Line(rng.gen()),
            11 => Operation::Circle(rng.gen()),
            12 => Operation::Ellipse(rng.gen()),
            13 => Operation::Rectangle(rng.gen()),
            _ => unreachable!(),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Push {
    width: Variable,
    height: Variable,
}

impl Push {
    fn apply(&self, context: &mut Context) -> bool {
        let width = context.scale_x(self.width.value(context)) as u32;
        let height = context.scale_y(self.height.value(context)) as u32;
        if width == 0 || height == 0 {
            return false;
        }
        context.frames.push(Frame::new(width, height));
        true
    }
}

impl Distribution<Push> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Push {
        Push {
            width: Variable::random(rng),
            height: Variable::random(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Pop {
    x: Variable,
    y: Variable,
}

impl Pop {
    fn apply(&self, context: &mut Context) -> bool {
        if context.frames.len() == 1 {
            return false;
        }
        let popped = context.frames.pop().unwrap();
        let x = context.scale_x(self.x.value(context));
        let y = context.scale_y(self.y.value(context));
        imageops::overlay(context.image_mut(), &popped.image, x, y);
        true
    }
}

impl Distribution<Pop> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Pop {
        Pop {
            x: Variable::random(rng),
            y: Variable::random(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Jump {
    target: Variable,
}

impl Jump {
    fn apply(&self, context: &mut Context) -> bool {
        context.program_counter = self.target.value(context) as usize;
        true
    }
}

impl Distribution<Jump> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Jump {
        Jump {
            target: Variable::random(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct BranchNotEqualZero {
    condition: Variable,
    target: Variable,
}

impl BranchNotEqualZero {
    fn apply(&self, context: &mut Context) -> bool {
        if self.condition.value(context) != 0 {
            context.program_counter = self.target.value(context) as usize;
        }
        true
    }
}

impl Distribution<BranchNotEqualZero> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BranchNotEqualZero {
        BranchNotEqualZero {
            target: Variable::random(rng),
            condition: Variable::random_non_constant(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Set {
    source: Variable,
    destination: Variable,
}

impl Set {
    fn apply(&self, context: &mut Context) -> bool {
        if !self.destination.can_mut() {
            return false;
        }
        *self.destination.value_mut(context) = self.source.value(context);
        true
    }
}

impl Distribution<Set> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Set {
        Set {
            source: Variable::random(rng),
            destination: Variable::random_register(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Add {
    source: Variable,
    destination: Variable,
}

impl Add {
    fn apply(&self, context: &mut Context) -> bool {
        if !self.destination.can_mut() {
            return false;
        }
        *self.destination.value_mut(context) = self
            .destination
            .value(context)
            .wrapping_add(self.source.value(context));

        true
    }
}

impl Distribution<Add> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Add {
        Add {
            source: Variable::random(rng),
            destination: Variable::random_register(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Subtract {
    source: Variable,
    destination: Variable,
}

impl Subtract {
    fn apply(&self, context: &mut Context) -> bool {
        if !self.destination.can_mut() {
            return false;
        }
        *self.destination.value_mut(context) = self
            .destination
            .value(context)
            .wrapping_sub(self.source.value(context));
        true
    }
}

impl Distribution<Subtract> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Subtract {
        Subtract {
            source: Variable::random(rng),
            destination: Variable::random_register(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Multiply {
    source: Variable,
    destination: Variable,
}

impl Multiply {
    fn apply(&self, context: &mut Context) -> bool {
        if !self.destination.can_mut() {
            return false;
        }
        *self.destination.value_mut(context) = self
            .destination
            .value(context)
            .wrapping_mul(self.source.value(context));
        true
    }
}

impl Distribution<Multiply> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Multiply {
        Multiply {
            source: Variable::random(rng),
            destination: Variable::random_register(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Size {
    width: Variable,
    height: Variable,
}

impl Size {
    fn apply(&self, context: &mut Context) -> bool {
        if context.frames.len() == 1 {
            // No scaling the base frame.
            return false;
        }
        let width = context.scale_x(self.width.value(context)) as u32;
        let height = context.scale_y(self.height.value(context)) as u32;
        if width == 0 || height == 0 {
            return false;
        }
        let mut buf = image::RgbaImage::new(width, height);
        imageops::replace(&mut buf, &context.frames.last().unwrap().image, 0, 0);
        *context.image_mut() = buf;
        true
    }
}

impl Distribution<Size> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Size {
        Size {
            width: Variable::random(rng),
            height: Variable::random(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Scale {
    width: Variable,
    height: Variable,
    smooth: Variable,
}

impl Scale {
    fn apply(&self, context: &mut Context) -> bool {
        if context.frames.len() == 1 {
            // No scaling the base frame.
            return false;
        }
        let width = self.width.value(context) as u32;
        let height = self.height.value(context) as u32;
        if width == 0 || height == 0 {
            return false;
        }
        let filter = if self.smooth.value(context) == 0 {
            imageops::FilterType::Nearest
        } else {
            imageops::FilterType::Gaussian
        };
        *context.image_mut() = imageops::resize(context.image(), width, height, filter);
        true
    }
}

impl Distribution<Scale> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Scale {
        Scale {
            width: Variable::random(rng),
            height: Variable::random(rng),
            smooth: Variable::random_constant(0, 1, rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Blur {
    amount: Variable,
}

impl Blur {
    fn apply(&self, context: &mut Context) -> bool {
        let sigma = self.amount.f32_value(context) * 3.0;
        imageops::blur(context.image_mut(), sigma);
        true
    }
}

impl Distribution<Blur> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Blur {
        Blur {
            amount: Variable::random(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Pixel {
    x: Variable,
    y: Variable,
    hue: Variable,
    saturation: Variable,
    value: Variable,
    alpha: Variable,
}

fn hsva_to_rgba(
    hue: &Variable,
    saturation: &Variable,
    value: &Variable,
    alpha: &Variable,
    context: &Context,
) -> image::Rgba<u8> {
    let hsva = palette::Hsva::new(
        hue.f32_value(context) * 360.0,
        saturation.f32_value(context),
        value.f32_value(context),
        alpha.f32_value(context),
    );
    let rgba: palette::Srgba = hsva.into_color();
    let components = rgba.into_components();
    return image::Rgba([
        (components.0 * 255.0) as u8,
        (components.1 * 255.0) as u8,
        (components.2 * 255.0) as u8,
        (components.3 * 255.0) as u8,
    ]);
}

impl Pixel {
    fn apply(&self, context: &mut Context) -> bool {
        let x = context.scale_x(self.x.value(context));
        let y = context.scale_y(self.y.value(context));
        let image = context.image();
        if x >= image.width() || y >= image.height() {
            return false;
        }

        let color = hsva_to_rgba(
            &self.hue,
            &self.saturation,
            &self.value,
            &self.alpha,
            context,
        );

        context.image_mut().draw_pixel(x, y, color);
        true
    }
}

impl Distribution<Pixel> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Pixel {
        Pixel {
            x: Variable::random(rng),
            y: Variable::random(rng),
            hue: Variable::random(rng),
            saturation: Variable::random(rng),
            value: Variable::random(rng),
            alpha: Variable::random(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct ReadPixel {
    x: Variable,
    y: Variable,
    hue: Variable,
    saturation: Variable,
    value: Variable,
    alpha: Variable,
}

impl ReadPixel {
    fn apply(&self, context: &mut Context) -> bool {
        let x = context.scale_x(self.x.value(context));
        let y = context.scale_y(self.y.value(context));
        let image = context.image();
        if x >= image.width() || y >= image.height() {
            return false;
        }
        if !(self.hue.can_mut()
            && self.saturation.can_mut()
            && self.value.can_mut()
            && self.alpha.can_mut())
        {
            return false;
        }

        let pixel = image.get_pixel(x, y).0;
        let rgba = palette::Srgba::new(
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
            pixel[3] as f32 / 255.0,
        );
        let hsva: palette::Hsva = rgba.into_color();
        let (h, s, v, a) = hsva.into_components();

        *self.hue.value_mut(context) = ((Into::<f32>::into(h) / 360.0) * 255.0) as u8;
        *self.saturation.value_mut(context) = (s * 255.0) as u8;
        *self.value.value_mut(context) = (v * 255.0) as u8;
        *self.alpha.value_mut(context) = (a * 255.0) as u8;
        true
    }
}

impl Distribution<ReadPixel> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> ReadPixel {
        ReadPixel {
            x: Variable::random(rng),
            y: Variable::random(rng),
            hue: Variable::random_register(rng),
            saturation: Variable::random_register(rng),
            value: Variable::random_register(rng),
            alpha: Variable::random_register(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Line {
    x1: Variable,
    y1: Variable,
    x2: Variable,
    y2: Variable,
    hue: Variable,
    saturation: Variable,
    value: Variable,
    alpha: Variable,
}

impl Line {
    fn apply(&self, context: &mut Context) -> bool {
        let start = (
            context.scale_x(self.x1.value(context)) as f32,
            context.scale_x(self.y1.value(context)) as f32,
        );
        let end = (
            context.scale_x(self.x2.value(context)) as f32,
            context.scale_x(self.y2.value(context)) as f32,
        );
        let color = hsva_to_rgba(
            &self.hue,
            &self.saturation,
            &self.value,
            &self.alpha,
            context,
        );
        imageproc::drawing::draw_line_segment_mut(context.image_mut(), start, end, color);
        true
    }
}

impl Distribution<Line> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Line {
        Line {
            x1: Variable::random(rng),
            y1: Variable::random(rng),
            x2: Variable::random(rng),
            y2: Variable::random(rng),
            hue: Variable::random(rng),
            saturation: Variable::random(rng),
            value: Variable::random(rng),
            alpha: Variable::random(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Circle {
    x: Variable,
    y: Variable,
    radius: Variable,
    fill: Variable,
    hue: Variable,
    saturation: Variable,
    value: Variable,
    alpha: Variable,
}

impl Circle {
    fn apply(&self, context: &mut Context) -> bool {
        let center = (
            context.scale_x(self.x.value(context)) as i32,
            context.scale_y(self.y.value(context)) as i32,
        );
        let radius = context.scale(self.radius.value(context)) as i32;
        let color = hsva_to_rgba(
            &self.hue,
            &self.saturation,
            &self.value,
            &self.alpha,
            context,
        );
        if self.fill.bool_value(context) {
            imageproc::drawing::draw_filled_circle_mut(context.image_mut(), center, radius, color);
        } else {
            imageproc::drawing::draw_hollow_circle_mut(context.image_mut(), center, radius, color);
        }
        true
    }
}

impl Distribution<Circle> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Circle {
        Circle {
            x: Variable::random(rng),
            y: Variable::random(rng),
            radius: Variable::random(rng),
            fill: Variable::random(rng),
            hue: Variable::random(rng),
            saturation: Variable::random(rng),
            value: Variable::random(rng),
            alpha: Variable::random(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Ellipse {
    x: Variable,
    y: Variable,
    width: Variable,
    height: Variable,
    fill: Variable,
    hue: Variable,
    saturation: Variable,
    value: Variable,
    alpha: Variable,
}

impl Ellipse {
    fn apply(&self, context: &mut Context) -> bool {
        let center = (
            context.scale_x(self.x.value(context)) as i32,
            context.scale_y(self.y.value(context)) as i32,
        );
        let width = context.scale_x(self.width.value(context)) as i32;
        let height = context.scale_y(self.height.value(context)) as i32;
        if width == 0 || height == 0 || width > 215 || height > 215 {
            // width^2 * height^2 must not exceed i32
            return false;
        }
        let color = hsva_to_rgba(
            &self.hue,
            &self.saturation,
            &self.value,
            &self.alpha,
            context,
        );
        if self.fill.bool_value(context) {
            imageproc::drawing::draw_filled_ellipse_mut(
                context.image_mut(),
                center,
                width,
                height,
                color,
            );
        } else {
            imageproc::drawing::draw_hollow_ellipse_mut(
                context.image_mut(),
                center,
                width,
                height,
                color,
            );
        }
        true
    }
}

impl Distribution<Ellipse> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Ellipse {
        Ellipse {
            x: Variable::random(rng),
            y: Variable::random(rng),
            width: Variable::random(rng),
            height: Variable::random(rng),
            fill: Variable::random(rng),
            hue: Variable::random(rng),
            saturation: Variable::random(rng),
            value: Variable::random(rng),
            alpha: Variable::random(rng),
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
struct Rectangle {
    x: Variable,
    y: Variable,
    width: Variable,
    height: Variable,
    fill: Variable,
    hue: Variable,
    saturation: Variable,
    value: Variable,
    alpha: Variable,
}

impl Rectangle {
    fn apply(&self, context: &mut Context) -> bool {
        let width = context.scale_x(self.width.value(context));
        let height = context.scale_y(self.height.value(context));
        if width == 0 || height == 0 {
            return false;
        }
        let rect = Rect::at(
            context.scale_x(self.x.value(context)) as i32,
            context.scale_y(self.y.value(context)) as i32,
        )
        .of_size(width, height);
        let color = hsva_to_rgba(
            &self.hue,
            &self.saturation,
            &self.value,
            &self.alpha,
            context,
        );
        if self.fill.bool_value(context) {
            imageproc::drawing::draw_filled_rect_mut(context.image_mut(), rect, color);
        } else {
            imageproc::drawing::draw_hollow_rect_mut(context.image_mut(), rect, color);
        }
        true
    }
}

impl Distribution<Rectangle> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Rectangle {
        Rectangle {
            x: Variable::random(rng),
            y: Variable::random(rng),
            width: Variable::random(rng),
            height: Variable::random(rng),
            fill: Variable::random(rng),
            hue: Variable::random(rng),
            saturation: Variable::random(rng),
            value: Variable::random(rng),
            alpha: Variable::random(rng),
        }
    }
}

fn main() {
    struct ScoredProgram {
        program: Program,
        total_score: f32,
        score_samples: i32,
        score_to_save: f32,
        remaining: i32,
    }

    impl ScoredProgram {
        fn new(program: Program, score_to_save: f32, remaining: i32) -> Self {
            Self {
                program,
                total_score: 0.0,
                score_samples: 0,
                score_to_save,
                remaining,
            }
        }

        fn score(&self) -> f32 {
            if self.score_samples == 0 || self.remaining == 0 {
                return 0.0;
            }
            self.total_score / self.score_samples as f32
        }
    }

    thread::scope(|s| {
        for t in 0..4 {
            let lt = t;
            s.spawn(move |_| 'outer: loop {
                let mut rng = rand::thread_rng();
                let mut population = Vec::new();
                let mut saved = 0;
                let start = Instant::now();

                loop {
                    while population.len() < 500 {
                        population.push(ScoredProgram::new(rng.gen(), 0.45, 20));
                    }

                    let mut average_score = 0.0;
                    let mut maximum_score: f32 = f32::MIN;
                    let mut average_operations = 0.0;
                    let mut denominator = 0;
                    for scored_program in &mut population {
                        let score;
                        if scored_program.score_samples < 5 {
                            let mut process = Process::new(&scored_program.program);
                            score = process.run(TRAIN_RESOLUTION, 1000, Duration::from_secs(10));
                            scored_program.total_score += score;
                            scored_program.score_samples += 1;
                        } else {
                            score = scored_program.score();
                            average_score += score;
                            maximum_score = maximum_score.max(score);
                            average_operations += scored_program.program.operations.len() as f32;
                            denominator += 1;
                        }
                        if scored_program.remaining > 0 {
                            scored_program.remaining -= 1;
                        }
                    }

                    if denominator > 0 {
                        average_score = average_score / denominator as f32;
                        average_operations = average_operations / denominator as f32;

                        println!(
                            "{}, {:.3}, {:.3}, {:.3}",
                            lt, average_score, maximum_score, average_operations
                        );
                    }

                    population.sort_unstable_by(|a, b| b.score().partial_cmp(&a.score()).unwrap());

                    let best = &mut population[0];
                    if best.score_samples == 5 && best.score() > best.score_to_save {
                        let mut process = Process::new(&best.program);
                        process.run(TEST_RESOLUTION, 10000, Duration::from_secs(10));
                        let path = format!(
                            "output/{}-{:.2}",
                            SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_millis(),
                            best.score(),
                        );
                        process.save(path.to_owned());
                        common::impressionize(&process.context.frames[0].image)
                            .save(format!("{}-imp.png", path))
                            .unwrap();
                        best.score_to_save = best.score() + 0.05;
                        if best.remaining > 0 {
                            best.remaining = 100;
                        }

                        saved += 1;
                        if saved > 2 {
                            continue 'outer;
                        }
                    }

                    if population.len() > 1000 {
                        population.drain(1000..population.len());
                    }

                    for i in 0..10 {
                        population[i].remaining = population[i].remaining.max(15 - i as i32);
                    }

                    for _ in 0..10 {
                        let parent = &population[rng.gen_range(0..population.len().min(100))];
                        let mut program = parent.program.clone();
                        program.mutate();
                        let score_to_save = parent.score_to_save;
                        let remaining = parent.remaining.max(10);
                        population.push(ScoredProgram::new(program, score_to_save, remaining));
                    }

                    for _ in 0..10 {
                        let parent_a = &population[rng.gen_range(0..population.len().min(25))];
                        let parent_b = &population[rng.gen_range(0..population.len())];
                        let program = Program::combine(&parent_a.program, &parent_b.program);
                        let score_to_save = parent_a.score_to_save.max(parent_b.score_to_save);
                        let remaining = parent_a.remaining.max(parent_b.remaining);
                        population.push(ScoredProgram::new(program, score_to_save, remaining));
                    }

                    if start.elapsed().as_secs() > 60 * 30 {
                        continue 'outer;
                    }
                }
            });
        }
    })
    .unwrap();
}

// Returns entropy from 0 to 1.
fn calculate_entropy(image: &image::RgbaImage) -> f32 {
    let mut counts = [0; 256];
    let mut n = 0;
    for &value in image.iter() {
        counts[value as usize] += 1;
        n += 1;
    }
    let mut entropy = 0.0;
    for &count in &counts {
        if count == 0 {
            continue;
        }
        let p: f32 = (count as f32) / (n as f32);
        entropy -= p * p.log(2.0);
    }
    entropy / 8.0
}

// Returns opacity from 0 to 1.
fn calculate_opacity(image: &image::RgbaImage) -> f32 {
    let mut opacity: u32 = 0;
    for (_, _, pixel) in image.enumerate_pixels() {
        opacity += pixel.0[3] as u32;
    }
    opacity as f32 / image.pixels().len() as f32 / u8::MAX as f32
}

// Returns noise (average difference between two adjacent pixels) from 0 to 1.
fn calculate_noise(image: &image::RgbaImage) -> f32 {
    let mut diff = 0;
    let mut denom = 0;

    for y in 1..image.height() as i32 - 1 {
        for x in 1..image.width() as i32 - 1 {
            let pixel = image.get_pixel(x as u32, y as u32);
            for dy in -1..1i32 {
                for dx in -1..1i32 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let other = image.get_pixel((x + dx) as u32, (y + dy) as u32);

                    for i in 0..pixel.0.len() {
                        diff += (pixel.0[i] as i32 - other.0[i] as i32).abs();
                        denom += 1;
                    }
                }
            }
        }
    }

    diff as f32 / denom as f32 / u8::MAX as f32
}
