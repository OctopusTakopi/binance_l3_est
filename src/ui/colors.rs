//! Static color palettes for bid and ask order visualisation.

use eframe::egui::Color32;
use once_cell::sync::Lazy;

pub static BID_COLORS: Lazy<Vec<Color32>> = Lazy::new(|| {
    vec![
        Color32::from_rgb(222, 235, 247),
        Color32::from_rgb(204, 227, 245),
        Color32::from_rgb(158, 202, 225),
        Color32::from_rgb(129, 189, 231),
        Color32::from_rgb(107, 174, 214),
        Color32::from_rgb(78, 157, 202),
        Color32::from_rgb(49, 130, 189),
        Color32::from_rgb(33, 113, 181),
        Color32::from_rgb(16, 96, 168),
        Color32::from_rgb(8, 81, 156),
    ]
});

pub static ASK_COLORS: Lazy<Vec<Color32>> = Lazy::new(|| {
    vec![
        Color32::from_rgb(254, 230, 206),
        Color32::from_rgb(253, 216, 186),
        Color32::from_rgb(253, 174, 107),
        Color32::from_rgb(253, 159, 88),
        Color32::from_rgb(253, 141, 60),
        Color32::from_rgb(245, 126, 47),
        Color32::from_rgb(230, 85, 13),
        Color32::from_rgb(204, 75, 12),
        Color32::from_rgb(179, 65, 10),
        Color32::from_rgb(166, 54, 3),
    ]
});
