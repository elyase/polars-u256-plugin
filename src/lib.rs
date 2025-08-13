use polars::prelude::*;
use pyo3::prelude::*;
use ruint::aliases::U256;
use std::borrow::Cow;

mod expressions;

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

fn check_len32(bytes: &[u8]) -> bool {
    bytes.len() == 32
}

// ---- Builder helpers implemented with Polars chunked builders ----
fn binary_series_from_iter(
    name: &PlSmallStr,
    it: impl Iterator<Item = Option<[u8; 32]>>,
) -> Series {
    BinaryChunked::from_iter_options(name.clone(), it).into_series()
}

fn boolean_series_from_iter(
    name: &PlSmallStr,
    it: impl Iterator<Item = Option<bool>>,
) -> Series {
    BooleanChunked::from_iter_options(name.clone(), it).into_series()
}

fn string_series_from_iter(
    name: &PlSmallStr,
    it: impl Iterator<Item = Option<String>>,
) -> Series {
    StringChunked::from_iter_options(name.clone(), it).into_series()
}

fn broadcast_binary_pair<'a>(
    a: &'a BinaryChunked,
    b: &'a BinaryChunked,
) -> PolarsResult<(Cow<'a, BinaryChunked>, Cow<'a, BinaryChunked>)> {
    let la = a.len();
    let lb = b.len();
    if la == lb {
        return Ok((Cow::Borrowed(a), Cow::Borrowed(b)));
    }
    if la == 1 && lb > 1 {
        let s = a.clone().into_series().new_from_index(0, lb);
        let bc = s.binary()?.clone();
        return Ok((Cow::Owned(bc), Cow::Borrowed(b)));
    }
    if lb == 1 && la > 1 {
        let s = b.clone().into_series().new_from_index(0, la);
        let bc = s.binary()?.clone();
        return Ok((Cow::Borrowed(a), Cow::Owned(bc)));
    }
    polars_bail!(InvalidOperation: "cannot do a binary operation on columns of different lengths: got {} and {}", la, lb)
}

fn map_pair_binary_to_binary_series<F>(
    name: &PlSmallStr,
    a: &BinaryChunked,
    b: &BinaryChunked,
    mut f: F,
) -> Series
where
    F: FnMut(&[u8], &[u8]) -> Option<[u8; 32]>,
{
    let (a_bc, b_bc) = broadcast_binary_pair(a, b).unwrap();
    let iter = a_bc
        .into_iter()
        .zip(&*b_bc)
        .map(|(la, rb)| match (la, rb) {
            (Some(la), Some(rb)) if check_len32(la) && check_len32(rb) => f(la, rb),
            _ => None,
        });
    binary_series_from_iter(name, iter)
}

fn map_pair_binary_to_bool_series<F>(
    name: &PlSmallStr,
    a: &BinaryChunked,
    b: &BinaryChunked,
    mut f: F,
) -> Series
where
    F: FnMut(&[u8], &[u8]) -> Option<bool>,
{
    let (a_bc, b_bc) = broadcast_binary_pair(a, b).unwrap();
    let iter = a_bc
        .into_iter()
        .zip(&*b_bc)
        .map(|(la, rb)| match (la, rb) {
            (Some(la), Some(rb)) if check_len32(la) && check_len32(rb) => f(la, rb),
            _ => None,
        });
    boolean_series_from_iter(name, iter)
}

fn map_unary_binary_to_binary_series<F>(name: &PlSmallStr, a: &BinaryChunked, mut f: F) -> Series
where
    F: FnMut(&[u8]) -> Option<[u8; 32]>,
{
    let iter = a.into_iter().map(|la| match la {
        Some(la) if check_len32(la) => f(la),
        _ => None,
    });
    binary_series_from_iter(name, iter)
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
        return Some(if an { Less } else { Greater });
    }
    Some(a.cmp(b))
}

fn i256_to_i64_opt(bytes: &[u8]) -> Option<i64> {
    if bytes.len() != 32 {
        return None;
    }
    if !i256_is_negative(bytes) {
        let (mag, _) = i256_abs_u256(bytes);
        let limbs = mag.as_limbs();
        if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 && limbs[0] <= i64::MAX as u64 {
            return Some(limbs[0] as i64);
        }
        None
    } else {
        let (mag, _) = i256_abs_u256(bytes);
        let limbs = mag.as_limbs();
        if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 && limbs[0] <= (1u128 << 63) as u64 {
            if limbs[0] == (1u128 << 63) as u64 {
                Some(i64::MIN)
            } else {
                let v = limbs[0] as i128;
                Some((-v) as i64)
            }
        } else {
            None
        }
    }
}
