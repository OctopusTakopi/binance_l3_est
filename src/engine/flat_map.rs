
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct FlatMap<K: Ord + Copy + Debug, V> {
    data: Vec<(K, V)>,
}

impl<K: Ord + Copy + Debug, V> FlatMap<K, V> {
    pub fn new() -> Self {
        Self { data: Vec::with_capacity(1024) }
    }

    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match self.data.binary_search_by_key(key, |&(k, _)| k) {
            Ok(idx) => Some(&mut self.data[idx].1),
            Err(_) => None,
        }
    }

    #[inline]
    pub fn insert(&mut self, key: K, value: V) {
        match self.data.binary_search_by_key(&key, |&(k, _)| k) {
            Ok(idx) => self.data[idx].1 = value,
            Err(idx) => self.data.insert(idx, (key, value)),
        }
    }

    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self.data.binary_search_by_key(key, |&(k, _)| k) {
            Ok(idx) => Some(self.data.remove(idx).1),
            Err(_) => None,
        }
    }

    #[inline]
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.data.binary_search_by_key(key, |&(k, _)| k).is_ok()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    
    #[inline]
    pub fn split_off(&mut self, key: &K) -> Self {
        let idx = match self.data.binary_search_by_key(key, |&(k, _)| k) {
            Ok(i) => i,
            Err(i) => i,
        };
        let tail = self.data.split_off(idx);
        Self { data: tail }
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    #[inline]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&K, &V)> + DoubleEndedIterator {
        self.data.iter().map(|(k, v)| (k, v))
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = (&K, &mut V)> + DoubleEndedIterator {
        self.data.iter_mut().map(|(k, v)| (&*k, v))
    }

    #[inline]
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> + DoubleEndedIterator {
        self.data.iter().map(|(_, v)| v)
    }

    #[inline]
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> + DoubleEndedIterator {
        self.data.iter().map(|(k, _)| k)
    }
}
