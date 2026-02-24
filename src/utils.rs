//! Shared serialization/deserialization utilities and common type aliases.
#![allow(dead_code)]

use std::fmt;

use jiff::Timestamp;
use serde::{
    Deserialize, Deserializer, de,
    de::{Error, Visitor},
};

/// A stack-allocated string for short symbol names (e.g. "BTCUSDT", "DOGEUSDT").
/// Avoids heap allocation for all symbols that fit within 16 bytes.
pub type SymbolStr = smallstr::SmallString<[u8; 16]>;

// ── i64 from JSON string ───────────────────────────────────────────────────────

struct I64Visitor;

impl Visitor<'_> for I64Visitor {
    type Value = Option<i64>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a string containing an i64 number")
    }

    fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if s.is_empty() {
            Ok(Some(0))
        } else {
            Ok(Some(s.parse::<i64>().map_err(Error::custom)?))
        }
    }
}

/// Deserialize a JSON string (`"123"`) into `i64`. Empty string → `0`.
pub fn from_str_to_i64<'de, D>(deserializer: D) -> Result<i64, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer
        .deserialize_str(I64Visitor)
        .map(|value| value.unwrap_or(0))
}

// ── f64 from JSON string ───────────────────────────────────────────────────────

struct F64Visitor;

impl Visitor<'_> for F64Visitor {
    type Value = Option<f64>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a string containing an f64 number")
    }

    fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if s.is_empty() {
            Ok(None)
        } else {
            Ok(Some(s.parse::<f64>().map_err(Error::custom)?))
        }
    }
}

/// Deserialize a JSON string (`"1.23"`) into `f64`. Empty string → `0.0`.
pub fn from_str_to_f64<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer
        .deserialize_str(F64Visitor)
        .map(|value| value.unwrap_or(0.0))
}

// ── Option<f64> from optional JSON string ─────────────────────────────────────

struct OptionF64Visitor;

impl<'de> Visitor<'de> for OptionF64Visitor {
    type Value = Option<f64>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("an optional string containing an f64 number")
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: Error,
    {
        Ok(None)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(F64Visitor)
    }
}

/// Deserialize an optional JSON string into `Option<f64>`.
/// `null` → `None`; `""` → `None`; `"1.23"` → `Some(1.23)`.
pub fn from_str_to_f64_opt<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_option(OptionF64Visitor)
}

// ── bool from JSON string ──────────────────────────────────────────────────────

struct BoolVisitor;

impl Visitor<'_> for BoolVisitor {
    type Value = Option<bool>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a string containing \"true\" or \"false\"")
    }

    fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        if s.is_empty() {
            Ok(None)
        } else {
            Ok(Some(
                s.to_lowercase().parse::<bool>().map_err(Error::custom)?,
            ))
        }
    }
}

/// Deserialize a JSON string (`"true"` / `"false"`) into `bool`.
/// Empty string → `false`.
pub fn from_str_to_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer
        .deserialize_str(BoolVisitor)
        .map(|value| value.unwrap_or(false))
}

// ── Option<bool> from optional JSON string ────────────────────────────────────

struct OptionBoolVisitor;

impl<'de> Visitor<'de> for OptionBoolVisitor {
    type Value = Option<bool>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("an optional string containing \"true\" or \"false\"")
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: Error,
    {
        Ok(None)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(BoolVisitor)
    }
}

/// Deserialize an optional JSON string into `Option<bool>`.
pub fn from_str_to_bool_opt<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_option(OptionBoolVisitor)
}

// ── String case transforms ─────────────────────────────────────────────────────

/// Deserialize a string and convert it to uppercase in one step.
pub fn to_uppercase<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let s: &str = Deserialize::deserialize(deserializer)?;
    Ok(s.to_uppercase())
}

/// Deserialize a string and convert it to lowercase in one step.
pub fn to_lowercase<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let s: &str = Deserialize::deserialize(deserializer)?;
    Ok(s.to_lowercase())
}

// ── Timestamp helpers ──────────────────────────────────────────────────────────

/// Return the current Unix time in milliseconds.
pub fn get_timestamp() -> u64 {
    Timestamp::now().as_millisecond() as u64
}

/// Deserialize a UTC datetime string (ISO 8601 / RFC 3339 / `"YYYY-MM-DD HH:MM:SS"`)
/// into Unix milliseconds as `u64`.
pub fn from_utc_to_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    let ts: Timestamp = s
        .parse()
        .map_err(|e| de::Error::custom(format!("Failed to parse date: {s}. Error: {e}")))?;
    Ok(ts.as_millisecond() as u64)
}

// ── Suppress unused-import warnings ───────────────────────────────────────────
// The items below are used indirectly via the `pub use` re-export in types.rs,
// so Rust may flag them; allow them here.
#[allow(unused_imports)]
use rand::Rng as _;
#[allow(unused_imports)]
use std::fmt::Write as _;
#[allow(unused_imports)]
use std::future::Future as _;
