#![allow(clippy::not_unsafe_ptr_arg_deref)]
use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::{LazyLock, Mutex};

use polars_core::prelude::*;
#[cfg(test)]
use polars_ffi::version_0::import_series;
use polars_ffi::version_0::{export_series, CallerContext, SeriesExport};
use ruint::aliases::U256;

use polars_arrow::array::{BinaryArray, MutableBinaryArray, MutableUtf8Array};
use polars_arrow::datatypes::{ArrowDataType, Field as ArrowField};
use polars_arrow::ffi::{export_field_to_c, import_field_from_c};

static LAST_ERROR: LazyLock<Mutex<CString>> =
    LazyLock::new(|| Mutex::new(CString::new("no error").unwrap()));

fn set_last_error(msg: &str) {
    let _ = LAST_ERROR
        .lock()
        .map(|mut s| *s = CString::new(msg).unwrap_or_else(|_| CString::new("error").unwrap()));
}

fn binary_series(s: &Series) -> PolarsResult<&BinaryChunked> {
    s.binary()
}

fn check_len32(bytes: &[u8]) -> bool {
    bytes.len() == 32
}

fn map_pair_binary_to_binary<F>(
    a: &BinaryChunked,
    b: &BinaryChunked,
    mut f: F,
) -> MutableBinaryArray<i32>
where
    F: FnMut(&[u8], &[u8]) -> Option<[u8; 32]>,
{
    let mut builder = MutableBinaryArray::<i32>::new();
    for (la, rb) in a.into_iter().zip(b.into_iter()) {
        match (la, rb) {
            (Some(la), Some(rb)) if check_len32(la) && check_len32(rb) => {
                if let Some(out) = f(la, rb) {
                    builder.push(Some(out.as_slice()));
                } else {
                    builder.push::<&[u8]>(None);
                }
            }
            (Some(_), Some(_)) => {
                set_last_error("non-32-byte value in input");
                builder.push::<&[u8]>(None);
            }
            _ => builder.push::<&[u8]>(None),
        }
    }
    builder
}

fn map_pair_binary_to_bool<F>(
    a: &BinaryChunked,
    b: &BinaryChunked,
    mut f: F,
) -> polars_arrow::array::MutableBooleanArray
where
    F: FnMut(&[u8], &[u8]) -> Option<bool>,
{
    let mut builder = polars_arrow::array::MutableBooleanArray::new();
    for (la, rb) in a.into_iter().zip(b.into_iter()) {
        match (la, rb) {
            (Some(la), Some(rb)) if check_len32(la) && check_len32(rb) => {
                if let Some(v) = f(la, rb) {
                    builder.push(Some(v));
                } else {
                    builder.push_null();
                }
            }
            (Some(_), Some(_)) => builder.push_null(),
            _ => builder.push_null(),
        }
    }
    builder
}

fn map_unary_binary_to_binary<F>(a: &BinaryChunked, mut f: F) -> MutableBinaryArray<i32>
where
    F: FnMut(&[u8]) -> Option<[u8; 32]>,
{
    let mut builder = MutableBinaryArray::<i32>::new();
    for la in a.into_iter() {
        match la {
            Some(la) if check_len32(la) => match f(la) {
                Some(out) => builder.push(Some(out.as_slice())),
                None => builder.push::<&[u8]>(None),
            },
            Some(_) => {
                set_last_error("non-32-byte value in input");
                builder.push::<&[u8]>(None);
            }
            None => builder.push::<&[u8]>(None),
        }
    }
    builder
}

#[no_mangle]
pub extern "C" fn _polars_plugin_get_version() -> u32 {
    // Align with polars_ffi-provided versioning (major << 16 | minor)
    let (major, minor) = polars_ffi::get_version();
    ((major as u32) << 16) | minor as u32
}

#[no_mangle]
pub extern "C" fn _polars_plugin_get_last_error_message() -> *const c_char {
    LAST_ERROR.lock().unwrap().as_ptr() as *const c_char
}

// Utilities
fn try_as_binary_series(s: &Series) -> PolarsResult<&BinaryChunked> {
    s.binary()
}

fn u256_from_be32(slice: &[u8]) -> Result<U256, &'static str> {
    if slice.len() != 32 {
        return Err("expected 32-byte value");
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(slice);
    Ok(U256::from_be_bytes(arr))
}

fn u256_to_be32(v: &U256) -> [u8; 32] {
    v.to_be_bytes()
}

// -------- Signed i256 helpers (two's complement over 256 bits) --------
fn i256_is_negative(bytes: &[u8]) -> bool {
    !bytes.is_empty() && (bytes[0] & 0x80) != 0
}

fn i256_twos_complement(bytes: &[u8]) -> [u8; 32] {
    let mut arr = [0u8; 32];
    arr.copy_from_slice(bytes);
    let v = U256::from_be_bytes(arr);
    let inv = !v;
    let (res, _) = inv.overflowing_add(U256::from(1u8));
    res.to_be_bytes()
}

fn i256_abs_u256(bytes: &[u8]) -> (U256, bool) {
    // returns (magnitude as U256, is_negative)
    let neg = i256_is_negative(bytes);
    if neg {
        let abs = i256_twos_complement(bytes);
        (U256::from_be_bytes(abs), true)
    } else {
        let mut arr = [0u8; 32];
        arr.copy_from_slice(bytes);
        (U256::from_be_bytes(arr), false)
    }
}

fn i256_cmp_bytes(a: &[u8], b: &[u8]) -> Option<std::cmp::Ordering> {
    if a.len() != 32 || b.len() != 32 {
        return None;
    }
    let an = i256_is_negative(a);
    let bn = i256_is_negative(b);
    use std::cmp::Ordering::*;
    if an != bn {
        // negative < positive
        return Some(if an { Less } else { Greater });
    }
    // same sign: lexicographic compare works for both positive and negative
    Some(a.cmp(b))
}

fn i256_to_i64_opt(bytes: &[u8]) -> Option<i64> {
    if bytes.len() != 32 {
        return None;
    }
    if !i256_is_negative(bytes) {
        // positive: must be <= i64::MAX
        let (mag, _) = i256_abs_u256(bytes);
        let limbs = mag.as_limbs();
        if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 && limbs[0] <= i64::MAX as u64 {
            return Some(limbs[0] as i64);
        }
        None
    } else {
        // negative: magnitude must be <= 2^63
        let (mag, _) = i256_abs_u256(bytes);
        let limbs = mag.as_limbs();
        if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 && limbs[0] <= (1u128 << 63) as u64 {
            if limbs[0] == (1u128 << 63) as u64 {
                return Some(i64::MIN);
            } else {
                let v = limbs[0] as i128;
                return Some((-v) as i64);
            }
        }
        None
    }
}

// removed: unused helper import_two_binary_series

fn series_from_binary_builder(name: &str, builder: MutableBinaryArray<i32>) -> Series {
    let arr: BinaryArray<i32> = builder.into();
    let name = PlSmallStr::from_string(name.to_string());
    // SAFETY: dtype matches array
    unsafe {
        Series::_try_from_arrow_unchecked_with_md(
            name,
            vec![arr.boxed()],
            &ArrowDataType::Binary,
            None,
        )
        .unwrap()
    }
}

// ---------- u256_add (elementwise) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must match the number of inputs. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_add(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_add expects exactly 2 input columns");
        return;
    }

    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];

    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };

    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let ua = u256_from_be32(la).unwrap();
        let ub = u256_from_be32(rb).unwrap();
        let (sum, overflow) = ua.overflowing_add(ub);
        if overflow {
            set_last_error("u256 addition overflow");
            return None;
        }
        Some(u256_to_be32(&sum))
    });
    let out = series_from_binary_builder(s0.name(), builder);

    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_add(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    // Use first input name if available; otherwise, default
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_add".into())
    } else {
        "u256_add".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_from_hex (string -> binary) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 1. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_from_hex(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 1 {
        set_last_error("u256_from_hex expects 1 column");
        return;
    }

    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];

    // Accept string or binary
    let out = match s0.dtype() {
        DataType::String => {
            let ca = s0.str().unwrap();
            let mut builder = MutableBinaryArray::<i32>::new();
            for opt_s in ca.into_iter() {
                if let Some(s) = opt_s {
                    let s = s.trim();
                    let s = s.strip_prefix("0x").unwrap_or(s);
                    if s.len() > 64 {
                        builder.push::<&[u8]>(None);
                        continue;
                    }
                    let mut buf = [0u8; 32];
                    // pad left with zeros
                    let padded_len = s.len().div_ceil(2); // bytes
                    let start = 32 - padded_len;
                    // If odd length, prepend a '0' for hex decoding
                    let s_owned;
                    let s_use: &str = if s.len() % 2 == 1 {
                        s_owned = format!("0{s}");
                        &s_owned
                    } else {
                        s
                    };
                    match hex::decode(s_use) {
                        Ok(decoded) => {
                            if decoded.len() > 32 {
                                builder.push::<&[u8]>(None);
                            } else {
                                buf[start..start + decoded.len()].copy_from_slice(&decoded);
                                builder.push(Some(buf.as_slice()));
                            }
                        }
                        Err(_) => builder.push::<&[u8]>(None),
                    }
                } else {
                    builder.push::<&[u8]>(None);
                }
            }
            series_from_binary_builder(s0.name(), builder)
        }
        DataType::Binary => {
            let ca = s0.binary().unwrap();
            let mut builder = MutableBinaryArray::<i32>::new();
            for opt_b in ca.into_iter() {
                if let Some(b) = opt_b {
                    if b.len() == 32 {
                        builder.push(Some(b));
                    } else if b.len() < 32 {
                        let mut out = [0u8; 32];
                        out[32 - b.len()..].copy_from_slice(b);
                        builder.push(Some(out.as_slice()));
                    } else {
                        builder.push::<&[u8]>(None);
                    }
                } else {
                    builder.push::<&[u8]>(None);
                }
            }
            series_from_binary_builder(s0.name(), builder)
        }
        _ => {
            set_last_error("u256_from_hex expects String or Binary input");
            return;
        }
    };

    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_from_hex(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_from_hex".into())
    } else {
        "u256_from_hex".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_to_hex (binary -> string) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 1. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_to_hex(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 1 {
        set_last_error("u256_to_hex expects 1 column");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];

    let ca = match s0.binary() {
        Ok(ca) => ca,
        Err(e) => {
            set_last_error(&format!("input not binary: {e}"));
            return;
        }
    };

    let mut builder = MutableUtf8Array::<i32>::with_capacity(ca.len());
    for opt in ca.into_iter() {
        match opt {
            Some(bytes) => {
                if bytes.len() != 32 {
                    builder.push::<&str>(None);
                } else {
                    let s = hex::encode(bytes);
                    // prepend 0x for friendlier display
                    let prefixed = format!("0x{s}");
                    builder.push(Some(prefixed.as_str()));
                }
            }
            None => builder.push::<&str>(None),
        }
    }
    let arr = <MutableUtf8Array<i32> as Into<polars_arrow::array::Utf8Array<i32>>>::into(builder);

    let name = PlSmallStr::from_string(s0.name().to_string());
    let out = Series::_try_from_arrow_unchecked_with_md(
        name,
        vec![arr.boxed()],
        &ArrowDataType::Utf8,
        None,
    )
    .unwrap();

    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_to_hex(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_to_hex".into())
    } else {
        "u256_to_hex".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Utf8, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_sub (elementwise) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_sub(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_sub expects exactly 2 input columns");
        return;
    }

    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];

    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }

    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let ua = u256_from_be32(la).unwrap();
        let ub = u256_from_be32(rb).unwrap();
        let Some(diff) = ua.checked_sub(ub) else {
            set_last_error("u256 subtraction underflow");
            return None;
        };
        Some(u256_to_be32(&diff))
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

// ---------- Bitwise ops: and/or/xor/not/shl/shr ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_bitand(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_bitand expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let ua = u256_from_be32(la).unwrap();
        let ub = u256_from_be32(rb).unwrap();
        Some(u256_to_be32(&(ua & ub)))
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_bitand(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_bitand".into())
    } else {
        "u256_bitand".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_bitor(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_bitor expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let ua = u256_from_be32(la).unwrap();
        let ub = u256_from_be32(rb).unwrap();
        Some(u256_to_be32(&(ua | ub)))
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_bitor(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_bitor".into())
    } else {
        "u256_bitor".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_bitxor(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_bitxor expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let ua = u256_from_be32(la).unwrap();
        let ub = u256_from_be32(rb).unwrap();
        Some(u256_to_be32(&(ua ^ ub)))
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_bitxor(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_bitxor".into())
    } else {
        "u256_bitxor".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 1. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_bitnot(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 1 {
        set_last_error("u256_bitnot expects exactly 1 input column");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("input not binary: {e}"));
            return;
        }
    };
    let builder = map_unary_binary_to_binary(a, |la| {
        let ua = u256_from_be32(la).unwrap();
        let inv = ua ^ U256::MAX;
        Some(u256_to_be32(&inv))
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_bitnot(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_bitnot".into())
    } else {
        "u256_bitnot".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

fn u256_from_be32_to_u64(slice: &[u8]) -> Option<u64> {
    let v = u256_from_be32(slice).ok()?;
    let limbs = v.as_limbs();
    if limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0 {
        None
    } else {
        Some(limbs[0])
    }
}

#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_shl(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_shl expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("base not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("shift not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let ua = u256_from_be32(la).unwrap();
        let shift = u256_from_be32_to_u64(rb)?;
        if shift >= 256 {
            return Some([0u8; 32]);
        }
        let res = ua << (shift as usize);
        Some(u256_to_be32(&res))
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_shl(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_shl".into())
    } else {
        "u256_shl".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_shr(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_shr expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("base not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("shift not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let ua = u256_from_be32(la).unwrap();
        let shift = u256_from_be32_to_u64(rb)?;
        if shift >= 256 {
            return Some([0u8; 32]);
        }
        let res = ua >> (shift as usize);
        Some(u256_to_be32(&res))
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_shr(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_shr".into())
    } else {
        "u256_shr".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_sub(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_sub".into())
    } else {
        "u256_sub".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_eq (elementwise comparison -> bool) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_eq(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_eq expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];

    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }

    let builder = map_pair_binary_to_bool(a, b, |la, rb| Some(la == rb));
    let arr: polars_arrow::array::BooleanArray = builder.into();
    let name = PlSmallStr::from_string(s0.name().to_string());
    let out = Series::_try_from_arrow_unchecked_with_md(
        name,
        vec![arr.boxed()],
        &ArrowDataType::Boolean,
        None,
    )
    .unwrap();

    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_eq(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_eq".into())
    } else {
        "u256_eq".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Boolean, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_lt (elementwise comparison -> bool) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_lt(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_lt expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];

    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }

    let builder = map_pair_binary_to_bool(a, b, |la, rb| Some(la < rb));
    let arr: polars_arrow::array::BooleanArray = builder.into();
    let name = PlSmallStr::from_string(s0.name().to_string());
    let out = Series::_try_from_arrow_unchecked_with_md(
        name,
        vec![arr.boxed()],
        &ArrowDataType::Boolean,
        None,
    )
    .unwrap();

    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_lt(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_lt".into())
    } else {
        "u256_lt".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Boolean, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_le (elementwise comparison -> bool) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_le(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_le expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }
    let builder = map_pair_binary_to_bool(a, b, |la, rb| Some(la <= rb));
    let arr: polars_arrow::array::BooleanArray = builder.into();
    let name = PlSmallStr::from_string(s0.name().to_string());
    let out = Series::_try_from_arrow_unchecked_with_md(
        name,
        vec![arr.boxed()],
        &ArrowDataType::Boolean,
        None,
    )
    .unwrap();
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_le(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_le".into())
    } else {
        "u256_le".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Boolean, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_gt (elementwise comparison -> bool) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_gt(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_gt expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }
    let builder = map_pair_binary_to_bool(a, b, |la, rb| Some(la > rb));
    let arr: polars_arrow::array::BooleanArray = builder.into();
    let name = PlSmallStr::from_string(s0.name().to_string());
    let out = Series::_try_from_arrow_unchecked_with_md(
        name,
        vec![arr.boxed()],
        &ArrowDataType::Boolean,
        None,
    )
    .unwrap();
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_gt(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_gt".into())
    } else {
        "u256_gt".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Boolean, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_ge (elementwise comparison -> bool) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_ge(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_ge expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }
    let builder = map_pair_binary_to_bool(a, b, |la, rb| Some(la >= rb));
    let arr: polars_arrow::array::BooleanArray = builder.into();
    let name = PlSmallStr::from_string(s0.name().to_string());
    let out = Series::_try_from_arrow_unchecked_with_md(
        name,
        vec![arr.boxed()],
        &ArrowDataType::Boolean,
        None,
    )
    .unwrap();
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_ge(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_ge".into())
    } else {
        "u256_ge".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Boolean, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_mul (elementwise) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_mul(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_mul expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let ua = u256_from_be32(la).unwrap();
        let ub = u256_from_be32(rb).unwrap();
        let (prod, overflow) = ua.overflowing_mul(ub);
        if overflow {
            set_last_error("u256 multiplication overflow");
            return None;
        }
        Some(u256_to_be32(&prod))
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_mul(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_mul".into())
    } else {
        "u256_mul".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_div (elementwise) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_div(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_div expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let ua = u256_from_be32(la).unwrap();
        let ub = u256_from_be32(rb).unwrap();
        if ub == U256::from(0u8) {
            set_last_error("u256 division by zero");
            return None;
        }
        let quot = ua / ub;
        Some(u256_to_be32(&quot))
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_div(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_div".into())
    } else {
        "u256_div".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_mod (elementwise) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_mod(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_mod expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let ua = u256_from_be32(la).unwrap();
        let ub = u256_from_be32(rb).unwrap();
        if ub == U256::from(0u8) {
            set_last_error("u256 modulo by zero");
            return None;
        }
        let rem = ua % ub;
        Some(u256_to_be32(&rem))
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_mod(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_mod".into())
    } else {
        "u256_mod".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_pow (power/exponentiation) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_pow(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("u256_pow expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];

    let a = match try_as_binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("base not binary: {e}"));
            return;
        }
    };
    let b = match try_as_binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("exponent not binary: {e}"));
            return;
        }
    };

    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }

    let mut builder = MutableBinaryArray::<i32>::new();
    for (la, rb) in a.into_iter().zip(b.into_iter()) {
        match (la, rb) {
            (Some(la), Some(rb)) => {
                if la.len() != 32 || rb.len() != 32 {
                    set_last_error("non-32-byte value in input");
                    builder.push::<&[u8]>(None);
                    continue;
                }
                let base = u256_from_be32(la).unwrap();
                let exp = u256_from_be32(rb).unwrap();

                // For power operation, we need to be careful about large exponents
                // Ruint's pow expects a U256, but we want to limit computation
                if exp > U256::from(64u8) {
                    set_last_error("u256 exponent too large (limited to 64 for safety)");
                    builder.push::<&[u8]>(None);
                    continue;
                }

                let result = base.pow(exp);

                let tmp = u256_to_be32(&result);
                builder.push(Some(tmp.as_slice()));
            }
            _ => builder.push::<&[u8]>(None),
        }
    }
    let out = series_from_binary_builder(s0.name(), builder);

    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_pow(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_pow".into())
    } else {
        "u256_pow".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_sum (aggregation) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 1. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_sum(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 1 {
        set_last_error("u256_sum expects exactly 1 input column");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];

    let a = match try_as_binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("input not binary: {e}"));
            return;
        }
    };

    let mut sum = U256::from(0u8);
    let mut has_valid_values = false;

    for bytes in a.into_iter().flatten() {
        if bytes.len() != 32 {
            continue; // Skip invalid values
        }
        has_valid_values = true;
        let value = u256_from_be32(bytes).unwrap();
        let (new_sum, overflow) = sum.overflowing_add(value);
        if overflow {
            set_last_error("u256 sum overflow");
            let null_series = Series::new(s0.name().clone(), [Option::<Vec<u8>>::None]);
            let exported = export_series(&null_series);
            std::ptr::write(out_series, exported);
            return;
        }
        sum = new_sum;
    }

    // Return null if no valid values, otherwise return sum
    let result = if has_valid_values {
        let sum_bytes = u256_to_be32(&sum);
        Series::new(s0.name().clone(), [Some(sum_bytes.to_vec())])
    } else {
        Series::new(s0.name().clone(), [Option::<Vec<u8>>::None])
    };

    let exported = export_series(&result);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_sum(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_sum".into())
    } else {
        "u256_sum".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- u256_to_int (binary -> u64) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 1. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_u256_to_int(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 1 {
        set_last_error("u256_to_int expects 1 column");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];

    let ca = match s0.binary() {
        Ok(ca) => ca,
        Err(e) => {
            set_last_error(&format!("input not binary: {e}"));
            return;
        }
    };

    let mut values: Vec<Option<u64>> = Vec::with_capacity(ca.len());
    for opt in ca.into_iter() {
        match opt {
            Some(bytes) => {
                if bytes.len() != 32 {
                    values.push(None);
                } else {
                    let val = u256_from_be32(bytes).unwrap();
                    if val > U256::from(u64::MAX) {
                        values.push(None);
                    } else {
                        values.push(Some(val.as_limbs()[0]));
                    }
                }
            }
            None => values.push(None),
        }
    }
    let out = Series::new(s0.name().clone(), values);

    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_u256_to_int(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "u256_to_int".into())
    } else {
        "u256_to_int".into()
    };
    let field = ArrowField::new(name, ArrowDataType::UInt64, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- i256_from_hex (string/binary -> binary two's complement) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 1. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_i256_from_hex(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 1 {
        set_last_error("i256_from_hex expects 1 column");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];

    let out = match s0.dtype() {
        DataType::String => {
            let ca = s0.str().unwrap();
            let mut builder = MutableBinaryArray::<i32>::new();
            for opt_s in ca.into_iter() {
                if let Some(sin) = opt_s {
                    let mut s = sin.trim();
                    let neg = s.starts_with('-');
                    if neg { s = &s[1..]; }
                    let s = s.strip_prefix("0x").unwrap_or(s);
                    if s.len() > 64 { builder.push::<&[u8]>(None); continue; }
                    let mut buf = [0u8; 32];
                    let padded_len = s.len().div_ceil(2);
                    let start = 32 - padded_len;
                    let s_owned;
                    let s_use: &str = if s.len() % 2 == 1 { s_owned = format!("0{s}"); &s_owned } else { s };
                    match hex::decode(s_use) {
                        Ok(decoded) => {
                            if decoded.len() > 32 { builder.push::<&[u8]>(None); }
                            else {
                                buf[start..start + decoded.len()].copy_from_slice(&decoded);
                                if neg {
                                    let negb = i256_twos_complement(&buf);
                                    builder.push(Some(negb.as_slice()));
                                } else {
                                    builder.push(Some(buf.as_slice()));
                                }
                            }
                        }
                        Err(_) => builder.push::<&[u8]>(None),
                    }
                } else { builder.push::<&[u8]>(None); }
            }
            series_from_binary_builder(s0.name(), builder)
        }
        DataType::Binary => {
            let ca = s0.binary().unwrap();
            let mut builder = MutableBinaryArray::<i32>::new();
            for opt_b in ca.into_iter() {
                if let Some(b) = opt_b {
                    if b.len() == 32 { builder.push(Some(b)); }
                    else if b.len() < 32 {
                        let mut out = [0u8; 32];
                        out[32 - b.len()..].copy_from_slice(b);
                        builder.push(Some(out.as_slice()));
                    } else { builder.push::<&[u8]>(None); }
                } else { builder.push::<&[u8]>(None); }
            }
            series_from_binary_builder(s0.name(), builder)
        }
        _ => { set_last_error("i256_from_hex expects String or Binary input"); return; }
    };

    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_i256_from_hex(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f).map(|af| af.name).unwrap_or_else(|_| "i256_from_hex".into())
    } else { "i256_from_hex".into() };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- i256_to_hex (binary -> signed hex string) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 1. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_i256_to_hex(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 1 { set_last_error("i256_to_hex expects 1 column"); return; }
    let inputs = match polars_ffi::version_0::import_series_buffer(input_ptr as *mut SeriesExport, input_len) {
        Ok(v) => v,
        Err(e) => { set_last_error(&format!("import failed: {e}")); return; }
    };
    let s0 = &inputs[0];
    let ca = match s0.binary() { Ok(ca) => ca, Err(e) => { set_last_error(&format!("input not binary: {e}")); return; } };
    let mut builder = MutableUtf8Array::<i32>::with_capacity(ca.len());
    for opt in ca.into_iter() {
        if let Some(bytes) = opt {
            if bytes.len() != 32 { builder.push::<&str>(None); continue; }
            if i256_is_negative(bytes) {
                let (mag, _) = i256_abs_u256(bytes);
                let s = hex::encode(u256_to_be32(&mag));
                let prefixed = format!("-0x{s}");
                builder.push(Some(prefixed.as_str()));
            } else {
                let s = hex::encode(bytes);
                let prefixed = format!("0x{s}");
                builder.push(Some(prefixed.as_str()));
            }
        } else { builder.push::<&str>(None); }
    }
    let arr = <MutableUtf8Array<i32> as Into<polars_arrow::array::Utf8Array<i32>>>::into(builder);
    let name = PlSmallStr::from_string(s0.name().to_string());
    let out = Series::_try_from_arrow_unchecked_with_md(name, vec![arr.boxed()], &ArrowDataType::Utf8, None).unwrap();
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_i256_to_hex(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() { import_field_from_c(f).map(|af| af.name).unwrap_or_else(|_| "i256_to_hex".into()) } else { "i256_to_hex".into() };
    let field = ArrowField::new(name, ArrowDataType::Utf8, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- i256 arithmetic (add/sub/mul wrapping) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr` and `out_series` must be valid pointers provided by Polars.
/// - `input_len` must be 2. Called only by Polars with valid lifetimes.
pub unsafe extern "C" fn _polars_plugin_i256_add(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 { set_last_error("i256_add expects exactly 2 input columns"); return; }
    let inputs = match polars_ffi::version_0::import_series_buffer(input_ptr as *mut SeriesExport, input_len) { Ok(v) => v, Err(e) => { set_last_error(&format!("import inputs failed: {e}")); return; } };
    let s0 = &inputs[0]; let s1 = &inputs[1];
    let a = match binary_series(s0) { Ok(v) => v, Err(e) => { set_last_error(&format!("lhs not binary: {e}")); return; } }; let b = match binary_series(s1) { Ok(v) => v, Err(e) => { set_last_error(&format!("rhs not binary: {e}")); return; } };
    if a.len() != b.len() { set_last_error("length mismatch"); return; }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let ua = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(la); t });
        let ub = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(rb); t });
        let (sum, _) = ua.overflowing_add(ub);
        Some(sum.to_be_bytes())
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr` and `out_field` must be valid pointers provided by Polars.
/// - `fields_len` must match the number of fields. Called only by Polars.
pub unsafe extern "C" fn _polars_plugin_field_i256_add(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() { import_field_from_c(f).map(|af| af.name).unwrap_or_else(|_| "i256_add".into()) } else { "i256_add".into() };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

#[no_mangle]
/// # Safety
pub unsafe extern "C" fn _polars_plugin_i256_sub(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) { 
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 { set_last_error("i256_sub expects exactly 2 input columns"); return; }
    let inputs = match polars_ffi::version_0::import_series_buffer(input_ptr as *mut SeriesExport, input_len) { Ok(v) => v, Err(e) => { set_last_error(&format!("import inputs failed: {e}")); return; } };
    let s0 = &inputs[0]; let s1 = &inputs[1];
    let a = match binary_series(s0) { Ok(v) => v, Err(e) => { set_last_error(&format!("lhs not binary: {e}")); return; } }; let b = match binary_series(s1) { Ok(v) => v, Err(e) => { set_last_error(&format!("rhs not binary: {e}")); return; } };
    if a.len() != b.len() { set_last_error("length mismatch"); return; }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let ua = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(la); t });
        let ub = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(rb); t });
        let (diff, _) = ua.overflowing_sub(ub);
        Some(diff.to_be_bytes())
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
pub unsafe extern "C" fn _polars_plugin_field_i256_sub(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() { import_field_from_c(f).map(|af| af.name).unwrap_or_else(|_| "i256_sub".into()) } else { "i256_sub".into() };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

#[no_mangle]
/// # Safety
pub unsafe extern "C" fn _polars_plugin_i256_mul(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 { set_last_error("i256_mul expects exactly 2 input columns"); return; }
    let inputs = match polars_ffi::version_0::import_series_buffer(input_ptr as *mut SeriesExport, input_len) { Ok(v) => v, Err(e) => { set_last_error(&format!("import inputs failed: {e}")); return; } };
    let s0 = &inputs[0]; let s1 = &inputs[1];
    let a = match binary_series(s0) { Ok(v) => v, Err(e) => { set_last_error(&format!("lhs not binary: {e}")); return; } }; let b = match binary_series(s1) { Ok(v) => v, Err(e) => { set_last_error(&format!("rhs not binary: {e}")); return; } };
    if a.len() != b.len() { set_last_error("length mismatch"); return; }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let ua = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(la); t });
        let ub = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(rb); t });
        let (prod, _) = ua.overflowing_mul(ub);
        Some(prod.to_be_bytes())
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
pub unsafe extern "C" fn _polars_plugin_field_i256_mul(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() { import_field_from_c(f).map(|af| af.name).unwrap_or_else(|_| "i256_mul".into()) } else { "i256_mul".into() };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- i256 div/mod (signed) ----------
#[no_mangle]
/// # Safety
pub unsafe extern "C" fn _polars_plugin_i256_div(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 { set_last_error("i256_div expects exactly 2 input columns"); return; }
    let inputs = match polars_ffi::version_0::import_series_buffer(input_ptr as *mut SeriesExport, input_len) { Ok(v) => v, Err(e) => { set_last_error(&format!("import inputs failed: {e}")); return; } };
    let s0=&inputs[0]; let s1=&inputs[1];
    let a = match binary_series(s0) { Ok(v)=>v, Err(e)=>{ set_last_error(&format!("lhs not binary: {e}")); return; } };
    let b = match binary_series(s1) { Ok(v)=>v, Err(e)=>{ set_last_error(&format!("rhs not binary: {e}")); return; } };
    if a.len()!=b.len() { set_last_error("length mismatch"); return; }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        // signed division with trunc toward zero
        let (amag, aneg) = i256_abs_u256(la);
        let (bmag, bneg) = i256_abs_u256(rb);
        if bmag == U256::from(0u8) {
            set_last_error("i256 division by zero");
            return None;
        }
        let q = amag / bmag;
        if (aneg ^ bneg) && q != U256::from(0u8) {
            let inv = (!q).overflowing_add(U256::from(1u8)).0;
            Some(inv.to_be_bytes())
        } else {
            Some(q.to_be_bytes())
        }
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
pub unsafe extern "C" fn _polars_plugin_field_i256_div(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f)=fields.first() { import_field_from_c(f).map(|af|af.name).unwrap_or_else(|_|"i256_div".into()) } else { "i256_div".into() };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

#[no_mangle]
/// # Safety
pub unsafe extern "C" fn _polars_plugin_i256_mod(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len()!=2 { set_last_error("i256_mod expects exactly 2 input columns"); return; }
    let inputs = match polars_ffi::version_0::import_series_buffer(input_ptr as *mut SeriesExport, input_len) { Ok(v)=>v, Err(e)=>{ set_last_error(&format!("import inputs failed: {e}")); return; } };
    let s0=&inputs[0]; let s1=&inputs[1];
    let a = match binary_series(s0){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("lhs not binary: {e}")); return; } };
    let b = match binary_series(s1){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("rhs not binary: {e}")); return; } };
    if a.len()!=b.len(){ set_last_error("length mismatch"); return; }
    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        let (amag, aneg) = i256_abs_u256(la);
        let (bmag, _) = i256_abs_u256(rb);
        if bmag == U256::from(0u8) {
            set_last_error("i256 modulo by zero");
            return None;
        }
        let r = amag % bmag;
        if r == U256::from(0u8) { return Some(r.to_be_bytes()); }
        if aneg {
            let inv = (!r).overflowing_add(U256::from(1u8)).0;
            Some(inv.to_be_bytes())
        } else {
            Some(r.to_be_bytes())
        }
    });
    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
pub unsafe extern "C" fn _polars_plugin_field_i256_mod(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f)=fields.first(){ import_field_from_c(f).map(|af|af.name).unwrap_or_else(|_|"i256_mod".into()) } else { "i256_mod".into() };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- i256 Euclidean division/remainder ----------
#[no_mangle]
/// # Safety
/// Computes Euclidean quotient: q such that a = b*q + r with r in [0, |b|).
pub unsafe extern "C" fn _polars_plugin_i256_div_euclid(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("i256_div_euclid expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }

    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        if la.len() != 32 || rb.len() != 32 {
            return None;
        }
        let (amag, aneg) = i256_abs_u256(la);
        let (bmag, bneg) = i256_abs_u256(rb);
        if bmag == U256::from(0u8) {
            set_last_error("i256 euclid division by zero");
            return None;
        }
        let q_abs = amag / bmag;
        let r_abs = amag % bmag;

        // Truncating quotient in two's complement
        let q_trunc_bytes = if (aneg ^ bneg) && q_abs != U256::from(0u8) {
            (!q_abs).overflowing_add(U256::from(1u8)).0.to_be_bytes()
        } else {
            q_abs.to_be_bytes()
        };

        // If remainder is negative under trunc (aneg && r_abs != 0), adjust quotient
        if r_abs != U256::from(0u8) && aneg {
            let mut q_u = U256::from_be_bytes(q_trunc_bytes);
            if !bneg {
                // q_e = q_trunc - 1
                q_u = q_u.overflowing_sub(U256::from(1u8)).0;
            } else {
                // q_e = q_trunc + 1
                q_u = q_u.overflowing_add(U256::from(1u8)).0;
            }
            Some(q_u.to_be_bytes())
        } else {
            Some(q_trunc_bytes)
        }
    });

    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
pub unsafe extern "C" fn _polars_plugin_field_i256_div_euclid(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "i256_div_euclid".into())
    } else {
        "i256_div_euclid".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

#[no_mangle]
/// # Safety
/// Computes Euclidean remainder: r in [0, |b|) such that a = b*q + r.
pub unsafe extern "C" fn _polars_plugin_i256_rem_euclid(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 2 {
        set_last_error("i256_rem_euclid expects exactly 2 input columns");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(
        input_ptr as *mut SeriesExport,
        input_len,
    ) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("lhs not binary: {e}"));
            return;
        }
    };
    let b = match binary_series(s1) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("rhs not binary: {e}"));
            return;
        }
    };
    if a.len() != b.len() {
        set_last_error("length mismatch");
        return;
    }

    let builder = map_pair_binary_to_binary(a, b, |la, rb| {
        if la.len() != 32 || rb.len() != 32 {
            return None;
        }
        let (amag, aneg) = i256_abs_u256(la);
        let (bmag, _bneg) = i256_abs_u256(rb);
        if bmag == U256::from(0u8) {
            set_last_error("i256 euclid remainder by zero");
            return None;
        }
        let r_abs = amag % bmag;
        if r_abs == U256::from(0u8) {
            return Some(r_abs.to_be_bytes());
        }
        if aneg {
            // r_e = |b| - r_abs
            let r_e = bmag.overflowing_sub(r_abs).0;
            Some(r_e.to_be_bytes())
        } else {
            Some(r_abs.to_be_bytes())
        }
    });

    let out = series_from_binary_builder(s0.name(), builder);
    let exported = export_series(&out);
    std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
pub unsafe extern "C" fn _polars_plugin_field_i256_rem_euclid(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) {
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f) = fields.first() {
        import_field_from_c(f)
            .map(|af| af.name)
            .unwrap_or_else(|_| "i256_rem_euclid".into())
    } else {
        "i256_rem_euclid".into()
    };
    let field = ArrowField::new(name, ArrowDataType::Binary, true);
    let exported = export_field_to_c(&field);
    std::ptr::write(out_field, exported);
}

// ---------- i256 comparisons ----------
#[no_mangle]
/// # Safety
/// - `input_ptr`/`out_series` must be valid and provided by Polars.
/// - `input_len` must be 2; lifetimes managed by Polars.
pub unsafe extern "C" fn _polars_plugin_i256_lt(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
    out_series: *mut SeriesExport,
    _ctx: *const CallerContext,
) {
    let inputs = match polars_ffi::version_0::import_series_buffer(input_ptr as *mut SeriesExport, input_len) { Ok(v)=>v, Err(e)=>{ set_last_error(&format!("import inputs failed: {e}")); return; } };
    if inputs.len()!=2 { set_last_error("i256_lt expects exactly 2 input columns"); return; }
    let s0=&inputs[0]; let s1=&inputs[1];
    let a = match binary_series(s0){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("lhs not binary: {e}")); return; } };
    let b = match binary_series(s1){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("rhs not binary: {e}")); return; } };
    if a.len()!=b.len(){ set_last_error("length mismatch"); return; }
    let mut builder = polars_arrow::array::MutableBooleanArray::new();
    for (la, rb) in a.into_iter().zip(b.into_iter()){
        match (la, rb){
            (Some(la), Some(rb)) if la.len()==32 && rb.len()==32 => {
                let ord = i256_cmp_bytes(la, rb).unwrap();
                builder.push(Some(matches!(ord, std::cmp::Ordering::Less)));
            }
            (Some(_), Some(_)) => builder.push_null(),
            _ => builder.push_null(),
        }
    }
    let arr: polars_arrow::array::BooleanArray = builder.into();
    let name = PlSmallStr::from_string(s0.name().to_string());
    let out = Series::_try_from_arrow_unchecked_with_md(name, vec![arr.boxed()], &ArrowDataType::Boolean, None).unwrap();
    let exported = export_series(&out); std::ptr::write(out_series, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr`/`out_field` must be valid.
/// - `fields_len` must match provided fields.
pub unsafe extern "C" fn _polars_plugin_field_i256_lt(
    fields_ptr: *const polars_arrow::ffi::ArrowSchema,
    fields_len: usize,
    out_field: *mut polars_arrow::ffi::ArrowSchema,
    _kwargs_ptr: *const u8,
    _kwargs_len: usize,
) { let fields = std::slice::from_raw_parts(fields_ptr, fields_len); let name = if let Some(f)=fields.first(){ import_field_from_c(f).map(|af|af.name).unwrap_or_else(|_|"i256_lt".into()) } else { "i256_lt".into() }; let field = ArrowField::new(name, ArrowDataType::Boolean, true); let exported = export_field_to_c(&field); std::ptr::write(out_field, exported);} 

#[no_mangle]
/// # Safety
/// - `input_ptr`/`out` must be valid and provided by Polars.
/// - `input_len` must be 2.
pub unsafe extern "C" fn _polars_plugin_i256_le(
    input_ptr:*const SeriesExport, input_len:usize, _k:*const u8, _kl:usize, out:*mut SeriesExport, _c:*const CallerContext){
    let inputs = match polars_ffi::version_0::import_series_buffer(input_ptr as *mut SeriesExport, input_len){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("import inputs failed: {e}")); return; } };
    if inputs.len()!=2 { set_last_error("i256_le expects exactly 2 input columns"); return; }
    let s0=&inputs[0]; let s1=&inputs[1]; let a=match binary_series(s0){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("lhs not binary: {e}")); return; } }; let b=match binary_series(s1){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("rhs not binary: {e}")); return; } }; if a.len()!=b.len(){ set_last_error("length mismatch"); return; }
    let mut builder = polars_arrow::array::MutableBooleanArray::new();
    for (la,rb) in a.into_iter().zip(b.into_iter()){
        match (la,rb){
            (Some(la),Some(rb)) if la.len()==32 && rb.len()==32 => {
                let ord = i256_cmp_bytes(la, rb).unwrap();
                builder.push(Some(!matches!(ord, std::cmp::Ordering::Greater)));
            }
            (Some(_),Some(_))=>builder.push_null(),
            _=>builder.push_null()
        }
    }
    let arr: polars_arrow::array::BooleanArray = builder.into();
    let name=PlSmallStr::from_string(s0.name().to_string());
    let out_s=Series::_try_from_arrow_unchecked_with_md(name, vec![arr.boxed()], &ArrowDataType::Boolean, None).unwrap();
    let exported = export_series(&out_s); std::ptr::write(out, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr`/`out_field` must be valid.
/// - `fields_len` must match provided fields.
pub unsafe extern "C" fn _polars_plugin_field_i256_le(
    fields_ptr:*const polars_arrow::ffi::ArrowSchema,
    fields_len:usize,
    out_field:*mut polars_arrow::ffi::ArrowSchema,
    _k:*const u8,
    _kl:usize,
){
    let fields = std::slice::from_raw_parts(fields_ptr, fields_len);
    let name = if let Some(f)=fields.first(){ import_field_from_c(f).map(|af|af.name).unwrap_or_else(|_|"i256_le".into()) } else { "i256_le".into() };
    let field = ArrowField::new(name, ArrowDataType::Boolean, true);
    let exported = export_field_to_c(&field); std::ptr::write(out_field, exported);
}

#[no_mangle]
/// # Safety
/// - `input_ptr`/`out` must be valid and provided by Polars.
/// - `input_len` must be 2.
pub unsafe extern "C" fn _polars_plugin_i256_gt(
    input_ptr:*const SeriesExport, input_len:usize, _k:*const u8, _kl:usize, out:*mut SeriesExport, _c:*const CallerContext
){
    let inputs = match polars_ffi::version_0::import_series_buffer(input_ptr as *mut SeriesExport, input_len){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("import inputs failed: {e}")); return; } };
    if inputs.len()!=2 { set_last_error("i256_gt expects exactly 2 input columns"); return; }
    let s0=&inputs[0]; let s1=&inputs[1];
    let a = match binary_series(s0){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("lhs not binary: {e}")); return; } };
    let b = match binary_series(s1){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("rhs not binary: {e}")); return; } };
    if a.len()!=b.len(){ set_last_error("length mismatch"); return; }
    let mut builder = polars_arrow::array::MutableBooleanArray::new();
    for (la,rb) in a.into_iter().zip(b.into_iter()){
        match (la,rb){
            (Some(la), Some(rb)) if la.len()==32 && rb.len()==32 => {
                let ord = i256_cmp_bytes(la, rb).unwrap(); builder.push(Some(matches!(ord, std::cmp::Ordering::Greater)));
            }
            (Some(_),Some(_))=>builder.push_null(), _=>builder.push_null()
        }
    }
    let arr: polars_arrow::array::BooleanArray = builder.into(); let name=PlSmallStr::from_string(s0.name().to_string()); let out_s=Series::_try_from_arrow_unchecked_with_md(name, vec![arr.boxed()], &ArrowDataType::Boolean, None).unwrap(); let exported = export_series(&out_s); std::ptr::write(out, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr`/`out_field` must be valid.
/// - `fields_len` must match provided fields.
pub unsafe extern "C" fn _polars_plugin_field_i256_gt(
    fields_ptr:*const polars_arrow::ffi::ArrowSchema, fields_len:usize, out_field:*mut polars_arrow::ffi::ArrowSchema, _k:*const u8, _kl:usize
){ let fields=std::slice::from_raw_parts(fields_ptr, fields_len); let name= if let Some(f)=fields.first(){ import_field_from_c(f).map(|af|af.name).unwrap_or_else(|_|"i256_gt".into()) } else { "i256_gt".into() }; let field=ArrowField::new(name, ArrowDataType::Boolean, true); let exported=export_field_to_c(&field); std::ptr::write(out_field, exported);} 

#[no_mangle]
/// # Safety
/// - `input_ptr`/`out` must be valid and provided by Polars.
/// - `input_len` must be 2.
pub unsafe extern "C" fn _polars_plugin_i256_ge(
    input_ptr:*const SeriesExport, input_len:usize, _k:*const u8, _kl:usize, out:*mut SeriesExport, _c:*const CallerContext
){
    let inputs = match polars_ffi::version_0::import_series_buffer(input_ptr as *mut SeriesExport, input_len){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("import inputs failed: {e}")); return; } };
    if inputs.len()!=2 { set_last_error("i256_ge expects exactly 2 input columns"); return; }
    let s0=&inputs[0]; let s1=&inputs[1]; let a=match binary_series(s0){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("lhs not binary: {e}")); return; } }; let b=match binary_series(s1){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("rhs not binary: {e}")); return; } }; if a.len()!=b.len(){ set_last_error("length mismatch"); return; }
    let mut builder = polars_arrow::array::MutableBooleanArray::new(); for (la,rb) in a.into_iter().zip(b.into_iter()){ match (la,rb){ (Some(la),Some(rb)) if la.len()==32 && rb.len()==32 => { let ord=i256_cmp_bytes(la,rb).unwrap(); builder.push(Some(!matches!(ord, std::cmp::Ordering::Less))); } (Some(_),Some(_))=>builder.push_null(), _=>builder.push_null() } } let arr: polars_arrow::array::BooleanArray=builder.into(); let name=PlSmallStr::from_string(s0.name().to_string()); let out_s=Series::_try_from_arrow_unchecked_with_md(name, vec![arr.boxed()], &ArrowDataType::Boolean, None).unwrap(); let exported=export_series(&out_s); std::ptr::write(out, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr`/`out_field` must be valid.
/// - `fields_len` must match provided fields.
pub unsafe extern "C" fn _polars_plugin_field_i256_ge(
    fields_ptr:*const polars_arrow::ffi::ArrowSchema, fields_len:usize, out_field:*mut polars_arrow::ffi::ArrowSchema, _k:*const u8, _kl:usize
){ let fields=std::slice::from_raw_parts(fields_ptr, fields_len); let name= if let Some(f)=fields.first(){ import_field_from_c(f).map(|af|af.name).unwrap_or_else(|_|"i256_ge".into()) } else { "i256_ge".into() }; let field=ArrowField::new(name, ArrowDataType::Boolean, true); let exported=export_field_to_c(&field); std::ptr::write(out_field, exported);} 

#[no_mangle]
/// # Safety
/// - `input_ptr`/`out` must be valid and provided by Polars.
/// - `input_len` must be 2.
pub unsafe extern "C" fn _polars_plugin_i256_eq(
    input_ptr:*const SeriesExport, input_len:usize, _k:*const u8, _kl:usize, out:*mut SeriesExport, _c:*const CallerContext
){ let inputs=match polars_ffi::version_0::import_series_buffer(input_ptr as *mut SeriesExport, input_len){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("import inputs failed: {e}")); return; } }; if inputs.len()!=2 { set_last_error("i256_eq expects exactly 2 input columns"); return; } let s0=&inputs[0]; let s1=&inputs[1]; let a=match binary_series(s0){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("lhs not binary: {e}")); return; } }; let b=match binary_series(s1){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("rhs not binary: {e}")); return; } }; if a.len()!=b.len(){ set_last_error("length mismatch"); return; } let mut builder=polars_arrow::array::MutableBooleanArray::new(); for (la,rb) in a.into_iter().zip(b.into_iter()){ match (la,rb){ (Some(la),Some(rb)) if la.len()==32 && rb.len()==32 => builder.push(Some(la==rb)), (Some(_),Some(_))=>builder.push_null(), _=>builder.push_null() } } let arr:polars_arrow::array::BooleanArray=builder.into(); let name=PlSmallStr::from_string(s0.name().to_string()); let out_s=Series::_try_from_arrow_unchecked_with_md(name, vec![arr.boxed()], &ArrowDataType::Boolean, None).unwrap(); let exported=export_series(&out_s); std::ptr::write(out, exported); }

#[no_mangle]
/// # Safety
/// - `fields_ptr`/`out_field` must be valid.
/// - `fields_len` must match provided fields.
pub unsafe extern "C" fn _polars_plugin_field_i256_eq(
    fields_ptr:*const polars_arrow::ffi::ArrowSchema, fields_len:usize, out_field:*mut polars_arrow::ffi::ArrowSchema, _k:*const u8, _kl:usize
){ let fields=std::slice::from_raw_parts(fields_ptr, fields_len); let name= if let Some(f)=fields.first(){ import_field_from_c(f).map(|af|af.name).unwrap_or_else(|_|"i256_eq".into()) } else { "i256_eq".into() }; let field=ArrowField::new(name, ArrowDataType::Boolean, true); let exported=export_field_to_c(&field); std::ptr::write(out_field, exported);} 

// ---------- i256 sum (wrapping) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr`/`out` must be valid and provided by Polars.
/// - `input_len` must be 1; lifetimes managed by Polars.
pub unsafe extern "C" fn _polars_plugin_i256_sum(
    input_ptr: *const SeriesExport,
    input_len: usize,
    _k: *const u8,
    _kl: usize,
    out: *mut SeriesExport,
    _c: *const CallerContext,
) {
    let inp = std::slice::from_raw_parts(input_ptr, input_len);
    if inp.len() != 1 {
        set_last_error("i256_sum expects exactly 1 input column");
        return;
    }
    let inputs = match polars_ffi::version_0::import_series_buffer(input_ptr as *mut SeriesExport, input_len) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("import inputs failed: {e}"));
            return;
        }
    };
    let s0 = &inputs[0];
    let a = match binary_series(s0) {
        Ok(v) => v,
        Err(e) => {
            set_last_error(&format!("input not binary: {e}"));
            return;
        }
    };
    let mut acc = U256::from(0u8);
    for bytes in a.into_iter().flatten() {
        if bytes.len() != 32 {
            continue;
        }
        let u = U256::from_be_bytes({
            let mut t = [0u8; 32];
            t.copy_from_slice(bytes);
            t
        });
        let (new, _) = acc.overflowing_add(u);
        acc = new;
    }
    let res = u256_to_be32(&acc);
    let out_s = Series::new(s0.name().clone(), [Some(res.to_vec())]);
    let exported = export_series(&out_s);
    std::ptr::write(out, exported);
}

#[no_mangle]
/// # Safety
/// - `fields_ptr`/`out_field` must be valid pointers provided by Polars.
/// - `fields_len` must correspond to the number of input fields.
pub unsafe extern "C" fn _polars_plugin_field_i256_sum(
    fields_ptr:*const polars_arrow::ffi::ArrowSchema, fields_len:usize, out_field:*mut polars_arrow::ffi::ArrowSchema, _k:*const u8, _kl:usize
){ let fields=std::slice::from_raw_parts(fields_ptr, fields_len); let name= if let Some(f)=fields.first(){ import_field_from_c(f).map(|af|af.name).unwrap_or_else(|_|"i256_sum".into()) } else { "i256_sum".into() }; let field=ArrowField::new(name, ArrowDataType::Binary, true); let exported=export_field_to_c(&field); std::ptr::write(out_field, exported);} 

// ---------- i256_to_int (binary -> i64) ----------
#[no_mangle]
/// # Safety
/// - `input_ptr`/`out` must be valid and provided by Polars.
/// - `input_len` must be 1; lifetimes managed by Polars.
pub unsafe extern "C" fn _polars_plugin_i256_to_int(
    input_ptr:*const SeriesExport, input_len:usize, _k:*const u8, _kl:usize, out:*mut SeriesExport, _c:*const CallerContext
){ let inputs=match polars_ffi::version_0::import_series_buffer(input_ptr as *mut SeriesExport, input_len){ Ok(v)=>v, Err(e)=>{ set_last_error(&format!("import failed: {e}")); return; } }; if inputs.len()!=1 { set_last_error("i256_to_int expects 1 column"); return; } let s0=&inputs[0]; let ca=match s0.binary(){ Ok(ca)=>ca, Err(e)=>{ set_last_error(&format!("input not binary: {e}")); return; } }; let mut vals:Vec<Option<i64>>=Vec::with_capacity(ca.len()); for v in ca.into_iter(){ if let Some(bytes)=v { if bytes.len()!=32 { vals.push(None); } else { vals.push(i256_to_i64_opt(bytes)); } } else { vals.push(None); } } let out_s = Series::new(s0.name().clone(), vals); let exported=export_series(&out_s); std::ptr::write(out, exported); }

#[no_mangle]
/// # Safety
/// - `fields_ptr`/`out_field` must be valid pointers provided by Polars.
/// - `fields_len` must correspond to the number of input fields.
pub unsafe extern "C" fn _polars_plugin_field_i256_to_int(
    fields_ptr:*const polars_arrow::ffi::ArrowSchema, fields_len:usize, out_field:*mut polars_arrow::ffi::ArrowSchema, _k:*const u8, _kl:usize
){ let fields=std::slice::from_raw_parts(fields_ptr, fields_len); let name= if let Some(f)=fields.first(){ import_field_from_c(f).map(|af|af.name).unwrap_or_else(|_|"i256_to_int".into()) } else { "i256_to_int".into() }; let field=ArrowField::new(name, ArrowDataType::Int64, true); let exported=export_field_to_c(&field); std::ptr::write(out_field, exported);} 
