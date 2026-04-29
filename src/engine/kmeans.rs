//! Mini-batch K-Means clustering for order book quantity levels.

use rand::Rng;
use std::cmp::Ordering;

/// Upper bound on the number of clusters. All inline arrays are sized to this.
const MAX_K: usize = 16;

#[derive(Clone, Copy, Debug, Default)]
struct Point {
    qty: f64,
}

fn normalize(points: &mut [Point]) {
    if points.is_empty() {
        return;
    }
    let mut min_q = f64::MAX;
    let mut max_q = f64::MIN;
    for p in points.iter() {
        min_q = min_q.min(p.qty);
        max_q = max_q.max(p.qty);
    }
    let range_q = max_q - min_q;
    if range_q > 0.0 {
        let inv = 1.0 / range_q;
        for p in points.iter_mut() {
            p.qty = (p.qty - min_q) * inv;
        }
    }
}

pub struct MiniBatchKMeans {
    num_clusters: usize,
    batch_size: usize,
    max_iter: usize,
    centroids: [Point; MAX_K],
    centroids_ready: bool,
    points: Vec<Point>,
    labels: Vec<usize>,
    sort_buf: Vec<Point>,
}

impl MiniBatchKMeans {
    pub fn new(num_clusters: usize, batch_size: usize, max_iter: usize) -> Self {
        assert!(num_clusters <= MAX_K);
        Self {
            num_clusters,
            batch_size,
            max_iter,
            centroids: [Point::default(); MAX_K],
            centroids_ready: false,
            points: Vec::with_capacity(1024),
            labels: Vec::with_capacity(1024),
            sort_buf: Vec::with_capacity(1024),
        }
    }

    pub fn fit_iter<I, J>(&mut self, iter: I) -> &[usize]
    where
        I: Iterator<Item = (i64, J)>,
        J: Iterator<Item = f64>,
    {
        self.points.clear();
        for (_price, order_iter) in iter {
            for qty in order_iter {
                if qty > 0.0 {
                    self.points.push(Point { qty });
                }
            }
        }
        if self.points.is_empty() {
            self.labels.clear();
            return &self.labels;
        }

        normalize(&mut self.points);

        if !self.centroids_ready {
            self.initialize_centroids();
            self.centroids_ready = true;
        }

        let mut counts = [0u32; MAX_K];
        let mut sums = [0.0f64; MAX_K];
        let k = self.num_clusters;
        let n = self.points.len();
        let batch_sz = self.batch_size.min(n);
        let mut rng = rand::rng();

        for _ in 0..self.max_iter {
            counts[..k].fill(0);
            sums[..k].fill(0.0);
            for _ in 0..batch_sz {
                let idx = rng.random_range(0..n);
                let qty = self.points[idx].qty;
                let c = self.closest_centroid(qty);
                sums[c] += qty;
                counts[c] += 1;
            }
            for i in 0..k {
                if counts[i] > 0 {
                    let lr = 1.0 / counts[i] as f64;
                    self.centroids[i].qty =
                        (1.0 - lr) * self.centroids[i].qty + lr * (sums[i] / counts[i] as f64);
                }
            }
        }

        self.labels.clear();
        for p in &self.points {
            self.labels.push(self.closest_centroid(p.qty));
        }

        let mut centroid_order = [0usize; MAX_K];
        for (i, slot) in centroid_order.iter_mut().take(k).enumerate() {
            *slot = i;
        }
        centroid_order[..k].sort_unstable_by(|&a, &b| {
            self.centroids[a]
                .qty
                .partial_cmp(&self.centroids[b].qty)
                .unwrap_or(Ordering::Equal)
        });
        let mut label_map = [0usize; MAX_K];
        for (new_label, &old_label) in centroid_order[..k].iter().enumerate() {
            label_map[old_label] = new_label;
        }
        for label in self.labels.iter_mut() {
            *label = label_map[*label];
        }
        &self.labels
    }

    #[inline(always)]
    fn closest_centroid(&self, qty: f64) -> usize {
        let mut min_dist = f64::INFINITY;
        let mut min_idx = 0;
        for i in 0..self.num_clusters {
            let d = (qty - self.centroids[i].qty).abs();
            if d < min_dist {
                min_dist = d;
                min_idx = i;
            }
        }
        min_idx
    }

    fn initialize_centroids(&mut self) {
        self.sort_buf.clear();
        self.sort_buf.extend_from_slice(&self.points);
        self.sort_buf
            .sort_unstable_by(|a, b| a.qty.partial_cmp(&b.qty).unwrap_or(Ordering::Equal));
        let n = self.sort_buf.len();
        let k = self.num_clusters;
        let step = if k > 1 { (n - 1) / (k - 1) } else { 0 };
        for i in 0..k {
            let idx = (i * step).min(n - 1);
            self.centroids[i] = self.sort_buf[idx];
        }
        for i in n.min(k)..k {
            self.centroids[i] = self.sort_buf[0];
        }
    }
}

pub fn build_clustered_orders<I, J>(iter: I, labels: &[usize]) -> Vec<(i64, Vec<(f64, usize)>)>
where
    I: Iterator<Item = (i64, J)>,
    J: Iterator<Item = f64>,
{
    let mut out = Vec::with_capacity(200);
    let mut idx = 0;
    for (price, order_iter) in iter {
        let mut level_orders = Vec::new();
        for qty in order_iter {
            if qty > 0.0 && idx < labels.len() {
                level_orders.push((qty, labels[idx]));
                idx += 1;
            }
        }
        out.push((price, level_orders));
    }
    out
}
