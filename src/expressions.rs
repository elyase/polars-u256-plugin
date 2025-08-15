use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use ruint::aliases::U256;
// Use Polars chunked builders; avoid raw Arrow builders.

use crate::{
    map_pair_binary_to_binary_series, map_pair_binary_to_bool_series, map_unary_binary_to_binary_series,
    string_series_from_iter, binary_series_from_iter, u256_from_be32, u256_to_be32, i256_abs_u256,
    i256_is_negative, i256_twos_complement, i256_cmp_bytes, i256_to_i64_opt,
};

#[polars_expr(output_type=Binary)]
pub fn u256_from_hex(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 {
        polars_bail!(ComputeError: "u256_from_hex expects 1 column");
    }
    let s0 = &inputs[0];
    let out = match s0.dtype() {
        DataType::String => {
            let ca = s0.str()?;
            let name = s0.name();
            let iter = ca.into_iter().map(|opt_s| {
                opt_s.and_then(|s| {
                    let s = s.trim();
                    let s = s.strip_prefix("0x").unwrap_or(s);
                    if s.len() > 64 {
                        return None;
                    }
                    let padded_len = s.len().div_ceil(2);
                    let start = 32 - padded_len;
                    let s_owned;
                    let s_use: &str = if s.len() % 2 == 1 {
                        s_owned = format!("0{s}");
                        &s_owned
                    } else {
                        s
                    };
                    match hex::decode(s_use) {
                        Ok(decoded) if decoded.len() <= 32 => {
                            let mut buf = [0u8; 32];
                            buf[start..start + decoded.len()].copy_from_slice(&decoded);
                            Some(buf)
                        }
                        _ => None,
                    }
                })
            });
            binary_series_from_iter(name, iter)
        }
        DataType::Binary => {
            let ca = s0.binary()?;
            let name = s0.name();
            let iter = ca.into_iter().map(|opt_b| match opt_b {
                Some(b) if b.len() == 32 => {
                    let mut out = [0u8; 32];
                    out.copy_from_slice(b);
                    Some(out)
                }
                Some(b) if b.len() < 32 => {
                    let mut out = [0u8; 32];
                    out[32 - b.len()..].copy_from_slice(b);
                    Some(out)
                }
                _ => None,
            });
            binary_series_from_iter(name, iter)
        }
        _ => polars_bail!(ComputeError: "u256_from_hex expects String or Binary input"),
    };
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn u256_from_int(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 {
        polars_bail!(ComputeError: "u256_from_int expects 1 column");
    }
    let s0 = &inputs[0];
    let name = s0.name();
    match s0.dtype() {
        dt if dt.is_unsigned_integer() => {
            let s_cast = s0.cast(&DataType::UInt64)?;
            let ca = s_cast.u64()?;
            let iter = ca.into_iter().map(|opt| opt.map(|v| {
                let mut out = [0u8; 32];
                out[24..].copy_from_slice(&v.to_be_bytes());
                out
            }));
            Ok(binary_series_from_iter(name, iter))
        }
        dt if dt.is_signed_integer() => {
            let s_cast = s0.cast(&DataType::Int64)?;
            let ca = s_cast.i64()?;
            let iter = ca.into_iter().map(|opt| opt.and_then(|v| {
                if v < 0 { return None; }
                let mut out = [0u8; 32];
                out[24..].copy_from_slice(&(v as u64).to_be_bytes());
                Some(out)
            }));
            Ok(binary_series_from_iter(name, iter))
        }
        _ => polars_bail!(ComputeError: "u256_from_int expects integer input (signed/unsigned)"),
    }
}

#[polars_expr(output_type=String)]
pub fn u256_to_hex(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 { polars_bail!(ComputeError: "u256_to_hex expects 1 column"); }
    let s0 = &inputs[0];
    let ca = s0.binary()?;
    let name = s0.name();
    let iter = ca
        .into_iter()
        .map(|opt| match opt {
            Some(bytes) if bytes.len() == 32 => Some(format!("0x{}", hex::encode(bytes))),
            _ => None,
        });
    Ok(string_series_from_iter(name, iter))
}

#[polars_expr(output_type=Binary)]
pub fn u256_add(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 2 { polars_bail!(ComputeError: "u256_add expects exactly 2 input columns"); }
    let s0 = &inputs[0];
    let s1 = &inputs[1];
    let a = s0.binary()?; let b = s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a, b, |la, rb| {
        let ua = u256_from_be32(la).ok()?;
        let ub = u256_from_be32(rb).ok()?;
        let (sum, overflow) = ua.overflowing_add(ub);
        if overflow { return None; }
        Some(u256_to_be32(&sum))
    });
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn u256_sub(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 2 { polars_bail!(ComputeError: "u256_sub expects exactly 2 input columns"); }
    let s0 = &inputs[0]; let s1 = &inputs[1];
    let a = s0.binary()?; let b = s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a, b, |la, rb| {
        let ua = u256_from_be32(la).ok()?;
        let ub = u256_from_be32(rb).ok()?;
        let diff = ua.checked_sub(ub)?;
        Some(u256_to_be32(&diff))
    });
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn u256_mul(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 2 { polars_bail!(ComputeError: "u256_mul expects exactly 2 input columns"); }
    let s0 = &inputs[0]; let s1 = &inputs[1];
    let a = s0.binary()?; let b = s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a, b, |la, rb| {
        let ua = u256_from_be32(la).ok()?;
        let ub = u256_from_be32(rb).ok()?;
        let (prod, overflow) = ua.overflowing_mul(ub);
        if overflow { return None; }
        Some(u256_to_be32(&prod))
    });
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn u256_div(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 2 { polars_bail!(ComputeError: "u256_div expects exactly 2 input columns"); }
    let s0 = &inputs[0]; let s1 = &inputs[1];
    let a = s0.binary()?; let b = s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a, b, |la, rb| {
        let ua = u256_from_be32(la).ok()?;
        let ub = u256_from_be32(rb).ok()?;
        if ub == U256::from(0u8) { return None; }
        Some(u256_to_be32(&(ua / ub)))
    });
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn u256_mod(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 2 { polars_bail!(ComputeError: "u256_mod expects exactly 2 input columns"); }
    let s0 = &inputs[0]; let s1 = &inputs[1];
    let a = s0.binary()?; let b = s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a, b, |la, rb| {
        let ua = u256_from_be32(la).ok()?;
        let ub = u256_from_be32(rb).ok()?;
        if ub == U256::from(0u8) { return None; }
        Some(u256_to_be32(&(ua % ub)))
    });
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn u256_pow(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 2 { polars_bail!(ComputeError: "u256_pow expects exactly 2 input columns"); }
    let s0 = &inputs[0]; let s1 = &inputs[1];
    let a = s0.binary()?; let b = s1.binary()?;
    let name = s0.name();
    let iter = a.into_iter().zip(b.into_iter()).map(|(la, rb)| {
        match (la, rb) {
            (Some(la), Some(rb)) if la.len() == 32 && rb.len() == 32 => {
                let base = u256_from_be32(la).unwrap();
                let exp = u256_from_be32(rb).unwrap();
                if exp > U256::from(64u8) { None } else { Some(u256_to_be32(&base.pow(exp))) }
            }
            _ => None,
        }
    });
    Ok(binary_series_from_iter(name, iter))
}

#[polars_expr(output_type=Boolean)]
pub fn u256_eq(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 2 { polars_bail!(ComputeError: "u256_eq expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1];
    let a=s0.binary()?; let b=s1.binary()?;
    Ok(map_pair_binary_to_bool_series(s0.name(), a, b, |la,rb| Some(la==rb)))
}

#[polars_expr(output_type=Boolean)]
pub fn u256_lt(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "u256_lt expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1];
    let a=s0.binary()?; let b=s1.binary()?;
    Ok(map_pair_binary_to_bool_series(s0.name(), a, b, |la,rb| Some(la < rb)))
}

#[polars_expr(output_type=Boolean)]
pub fn u256_le(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "u256_le expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1];
    let a=s0.binary()?; let b=s1.binary()?;
    Ok(map_pair_binary_to_bool_series(s0.name(), a, b, |la,rb| Some(la <= rb)))
}

#[polars_expr(output_type=Boolean)]
pub fn u256_gt(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "u256_gt expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1];
    let a=s0.binary()?; let b=s1.binary()?;
    Ok(map_pair_binary_to_bool_series(s0.name(), a, b, |la,rb| Some(la > rb)))
}

#[polars_expr(output_type=Boolean)]
pub fn u256_ge(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "u256_ge expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1];
    let a=s0.binary()?; let b=s1.binary()?;
    Ok(map_pair_binary_to_bool_series(s0.name(), a, b, |la,rb| Some(la >= rb)))
}

#[polars_expr(output_type=Binary)]
pub fn u256_shl(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "u256_shl expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1]; let a=s0.binary()?; let b=s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a,b, |la,rb| {
        let ua = u256_from_be32(la).ok()?;
        // interpret rb as shift amount u64
        let binding = u256_from_be32(rb).ok()?;
        let limbs = binding.as_limbs();
        if limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0 { return Some([0u8;32]); }
        let sft = limbs[0]; if sft >= 256 { return Some([0u8;32]); }
        Some(u256_to_be32(&(ua << (sft as usize))))
    });
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn u256_shr(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "u256_shr expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1]; let a=s0.binary()?; let b=s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a,b, |la,rb| {
        let ua = u256_from_be32(la).ok()?;
        // interpret rb as shift amount u64
        let binding = u256_from_be32(rb).ok()?;
        let limbs = binding.as_limbs();
        if limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0 { return Some([0u8;32]); }
        let sft = limbs[0]; if sft >= 256 { return Some([0u8;32]); }
        Some(u256_to_be32(&(ua >> (sft as usize))))
    });
    Ok(out)
}

#[polars_expr(output_type=UInt64)]
pub fn u256_to_int(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=1 { polars_bail!(ComputeError: "u256_to_int expects 1 column"); }
    let s0=&inputs[0]; let ca=s0.binary()?; let mut vals:Vec<Option<u64>>=Vec::with_capacity(ca.len());
    for v in ca.into_iter(){ if let Some(bytes)=v { if bytes.len()!=32 { vals.push(None); } else {
            let v = u256_from_be32(bytes).unwrap();
            let limbs = v.as_limbs();
            if limbs[1]==0 && limbs[2]==0 && limbs[3]==0 { vals.push(Some(limbs[0])); } else { vals.push(None); }
        } } else { vals.push(None); } }
    Ok(Series::new(s0.name().clone(), vals))
}

// -------------------- i256 helpers exposed as expressions --------------------

#[polars_expr(output_type=Binary)]
pub fn i256_from_hex(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=1 { polars_bail!(ComputeError: "i256_from_hex expects 1 column"); }
    let s0 = &inputs[0];
    let out = match s0.dtype(){
        DataType::String => {
            let ca = s0.str()?;
            let name = s0.name();
            let iter = ca.into_iter().map(|opt| opt.and_then(|sin| {
                let mut s = sin.trim();
                let neg = s.starts_with('-');
                if neg { s = &s[1..]; }
                let s = s.strip_prefix("0x").unwrap_or(s);
                if s.len() > 64 { return None; }
                let padded = s.len().div_ceil(2);
                let start = 32 - padded;
                let s_owned;
                let s_use: &str = if s.len() % 2 == 1 { s_owned = format!("0{s}"); &s_owned } else { s };
                match hex::decode(s_use) {
                    Ok(decoded) if decoded.len() <= 32 => {
                        let mut buf = [0u8; 32];
                        buf[start..start + decoded.len()].copy_from_slice(&decoded);
                        if neg { Some(i256_twos_complement(&buf)) } else { Some(buf) }
                    }
                    _ => None,
                }
            }));
            binary_series_from_iter(name, iter)
        }
        DataType::Binary => {
            let ca=s0.binary()?; let name = s0.name();
            let iter = ca.into_iter().map(|opt| match opt {
                Some(b) if b.len()==32 => { let mut out=[0u8;32]; out.copy_from_slice(b); Some(out) }
                Some(b) if b.len()<32 => { let mut out=[0u8;32]; out[32-b.len()..].copy_from_slice(b); Some(out) }
                _ => None,
            });
            binary_series_from_iter(name, iter)
        }
        _ => polars_bail!(ComputeError: "i256_from_hex expects String or Binary input"),
    };
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn i256_from_int(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=1 { polars_bail!(ComputeError: "i256_from_int expects 1 column"); }
    let s0=&inputs[0]; let name = s0.name();
    match s0.dtype() {
        dt if dt.is_unsigned_integer() => {
            let s_cast = s0.cast(&DataType::UInt64)?;
            let ca = s_cast.u64()?;
            let iter = ca.into_iter().map(|opt| opt.map(|v| {
                let mut out=[0u8;32]; out[24..].copy_from_slice(&v.to_be_bytes()); out
            }));
            Ok(binary_series_from_iter(name, iter))
        }
        dt if dt.is_signed_integer() => {
            let s_cast = s0.cast(&DataType::Int64)?;
            let ca = s_cast.i64()?;
            let iter = ca.into_iter().map(|opt| opt.map(|v| {
                if v >= 0 {
                    let mut out=[0u8;32]; out[24..].copy_from_slice(&(v as u64).to_be_bytes()); out
                } else {
                    let mag = (-v) as u64;
                    let res = U256::from(0u8).overflowing_sub(U256::from(mag)).0;
                    res.to_be_bytes()
                }
            }));
            Ok(binary_series_from_iter(name, iter))
        }
        _ => polars_bail!(ComputeError: "i256_from_int expects integer input"),
    }
}

#[polars_expr(output_type=String)]
pub fn i256_to_hex(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=1 { polars_bail!(ComputeError: "i256_to_hex expects 1 column"); }
    let s0=&inputs[0]; let ca=s0.binary()?; let name = s0.name();
    let iter = ca.into_iter().map(|opt| match opt {
        Some(bytes) if bytes.len()==32 => {
            if i256_is_negative(bytes) {
                let (mag,_) = i256_abs_u256(bytes);
                Some(format!("-0x{}", hex::encode(u256_to_be32(&mag))))
            } else {
                Some(format!("0x{}", hex::encode(bytes)))
            }
        }
        _ => None,
    });
    Ok(string_series_from_iter(name, iter))
}

#[polars_expr(output_type=Binary)]
pub fn i256_mod(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "i256_mod expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1]; let a=s0.binary()?; let b=s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a,b, |la,rb| {
        let (amag, aneg) = i256_abs_u256(la);
        let (bmag, _bneg) = i256_abs_u256(rb);
        if bmag == U256::from(0u8) { return None; }
        let r = amag % bmag;
        if r == U256::from(0u8) { return Some(r.to_be_bytes()); }
        if aneg { let inv = (!r).overflowing_add(U256::from(1u8)).0; Some(inv.to_be_bytes()) } else { Some(r.to_be_bytes()) }
    });
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn i256_div_euclid(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "i256_div_euclid expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1]; let a=s0.binary()?; let b=s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a,b, |la,rb| {
        let (amag, aneg) = i256_abs_u256(la);
        let (bmag, bneg) = i256_abs_u256(rb);
        if bmag == U256::from(0u8) { return None; }
        let q_abs = amag / bmag;
        let r_abs = amag % bmag;
        let q_trunc = if (aneg ^ bneg) && q_abs != U256::from(0u8) { (!q_abs).overflowing_add(U256::from(1u8)).0 } else { q_abs };
        if r_abs != U256::from(0u8) && aneg {
            let mut q_u = q_trunc;
            if !bneg { q_u = q_u.overflowing_sub(U256::from(1u8)).0; } else { q_u = q_u.overflowing_add(U256::from(1u8)).0; }
            Some(q_u.to_be_bytes())
        } else { Some(q_trunc.to_be_bytes()) }
    });
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn i256_rem_euclid(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "i256_rem_euclid expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1]; let a=s0.binary()?; let b=s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a,b, |la,rb| {
        let (amag, aneg) = i256_abs_u256(la);
        let (bmag, _bneg) = i256_abs_u256(rb);
        if bmag == U256::from(0u8) { return None; }
        let r_abs = amag % bmag;
        if r_abs == U256::from(0u8) { return Some(r_abs.to_be_bytes()); }
        if aneg {
            // trunc remainder is negative: r_t = -r_abs, Euclidean remainder: r_e = bmag - r_abs
            Some((bmag - r_abs).to_be_bytes())
        } else { Some(r_abs.to_be_bytes()) }
    });
    Ok(out)
}

#[polars_expr(output_type=Boolean)]
pub fn i256_eq(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "i256_eq expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1]; let a=s0.binary()?; let b=s1.binary()?;
    Ok(map_pair_binary_to_bool_series(s0.name(), a,b, |la,rb| Some(la==rb)))
}

#[polars_expr(output_type=Boolean)]
pub fn i256_lt(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "i256_lt expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1]; let a=s0.binary()?; let b=s1.binary()?;
    Ok(map_pair_binary_to_bool_series(s0.name(), a,b, |la,rb| i256_cmp_bytes(la,rb).map(|o| o.is_lt())))
}

#[polars_expr(output_type=Boolean)]
pub fn i256_le(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "i256_le expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1]; let a=s0.binary()?; let b=s1.binary()?;
    Ok(map_pair_binary_to_bool_series(s0.name(), a,b, |la,rb| i256_cmp_bytes(la,rb).map(|o| !o.is_gt())))
}

#[polars_expr(output_type=Boolean)]
pub fn i256_gt(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "i256_gt expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1]; let a=s0.binary()?; let b=s1.binary()?;
    Ok(map_pair_binary_to_bool_series(s0.name(), a,b, |la,rb| i256_cmp_bytes(la,rb).map(|o| o.is_gt())))
}

#[polars_expr(output_type=Boolean)]
pub fn i256_ge(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "i256_ge expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1]; let a=s0.binary()?; let b=s1.binary()?;
    Ok(map_pair_binary_to_bool_series(s0.name(), a,b, |la,rb| i256_cmp_bytes(la,rb).map(|o| !o.is_lt())))
}

#[polars_expr(output_type=Binary)]
pub fn i256_sum(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=1 { polars_bail!(ComputeError: "i256_sum expects exactly 1 input column"); }
    let s0=&inputs[0]; let a=s0.binary()?; let mut acc=U256::from(0u8); for bytes in a.into_iter().flatten(){ if bytes.len()!=32 { continue; } let u = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(bytes); t}); let (new,_) = acc.overflowing_add(u); acc=new; }
    let res = u256_to_be32(&acc); Ok(Series::new(s0.name().clone(), [Some(res.to_vec())]))
}

#[polars_expr(output_type=Int64)]
pub fn i256_to_int(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=1 { polars_bail!(ComputeError: "i256_to_int expects 1 column"); }
    let s0=&inputs[0]; let ca=s0.binary()?; let mut vals:Vec<Option<i64>>=Vec::with_capacity(ca.len());
    for v in ca.into_iter(){ if let Some(bytes)=v { if bytes.len()!=32 { vals.push(None); } else { vals.push(i256_to_i64_opt(bytes)); } } else { vals.push(None); } }
    Ok(Series::new(s0.name().clone(), vals))
}

#[polars_expr(output_type=Binary)]
pub fn u256_bitand(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "u256_bitand expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1];
    let a=s0.binary()?; let b=s1.binary()?;
    Ok(map_pair_binary_to_binary_series(s0.name(), a,b, |la,rb| { let ua=u256_from_be32(la).ok()?; let ub=u256_from_be32(rb).ok()?; Some(u256_to_be32(&(ua & ub))) }))
}

#[polars_expr(output_type=Binary)]
pub fn u256_bitor(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "u256_bitor expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1];
    let a=s0.binary()?; let b=s1.binary()?;
    Ok(map_pair_binary_to_binary_series(s0.name(), a,b, |la,rb| { let ua=u256_from_be32(la).ok()?; let ub=u256_from_be32(rb).ok()?; Some(u256_to_be32(&(ua | ub))) }))
}

#[polars_expr(output_type=Binary)]
pub fn u256_bitxor(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=2 { polars_bail!(ComputeError: "u256_bitxor expects exactly 2 input columns"); }
    let s0=&inputs[0]; let s1=&inputs[1];
    let a=s0.binary()?; let b=s1.binary()?;
    Ok(map_pair_binary_to_binary_series(s0.name(), a,b, |la,rb| { let ua=u256_from_be32(la).ok()?; let ub=u256_from_be32(rb).ok()?; Some(u256_to_be32(&(ua ^ ub))) }))
}

#[polars_expr(output_type=Binary)]
pub fn u256_bitnot(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len()!=1 { polars_bail!(ComputeError: "u256_bitnot expects exactly 1 input column"); }
    let s0=&inputs[0]; let a=s0.binary()?;
    Ok(map_unary_binary_to_binary_series(s0.name(), a, |la| { let ua=u256_from_be32(la).ok()?; Some(u256_to_be32(&(ua ^ U256::MAX))) }))
}

#[polars_expr(output_type=Binary)]
pub fn u256_sum(inputs: &[polars::prelude::Series]) -> polars::prelude::PolarsResult<polars::prelude::Series> {
    if inputs.len() != 1 { polars_bail!(ComputeError: "u256_sum expects exactly 1 input column"); }
    let s0 = &inputs[0];
    let a = s0.binary()?;
    let mut sum = U256::from(0u8);
    let mut has_valid = false;
    for bytes in a.into_iter().flatten() {
        if bytes.len() != 32 { continue; }
        has_valid = true;
        let v = u256_from_be32(bytes).unwrap();
        let (new_sum, overflow) = sum.overflowing_add(v);
        if overflow { return Ok(Series::new(s0.name().clone(), [Option::<Vec<u8>>::None])); }
        sum = new_sum;
    }
    let out = if has_valid {
        Series::new(s0.name().clone(), [Some(u256_to_be32(&sum).to_vec())])
    } else {
        Series::new(s0.name().clone(), [Option::<Vec<u8>>::None])
    };
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn i256_add(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 2 { polars_bail!(ComputeError: "i256_add expects exactly 2 input columns"); }
    let s0 = &inputs[0]; let s1 = &inputs[1];
    let a = s0.binary()?; let b = s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a, b, |la, rb| {
        let ua = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(la); t });
        let ub = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(rb); t });
        Some(ua.overflowing_add(ub).0.to_be_bytes())
    });
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn i256_sub(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 2 { polars_bail!(ComputeError: "i256_sub expects exactly 2 input columns"); }
    let s0 = &inputs[0]; let s1 = &inputs[1];
    let a = s0.binary()?; let b = s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a, b, |la, rb| {
        let ua = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(la); t });
        let ub = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(rb); t });
        Some(ua.overflowing_sub(ub).0.to_be_bytes())
    });
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn i256_div(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 2 { polars_bail!(ComputeError: "i256_div expects exactly 2 input columns"); }
    let s0 = &inputs[0]; let s1 = &inputs[1];
    let a = s0.binary()?; let b = s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a, b, |la, rb| {
        let (amag, aneg) = i256_abs_u256(la);
        let (bmag, bneg) = i256_abs_u256(rb);
        if bmag == U256::from(0u8) { return None; }
        let q = amag / bmag;
        if (aneg ^ bneg) && q != U256::from(0u8) {
            let inv = (!q).overflowing_add(U256::from(1u8)).0;
            Some(inv.to_be_bytes())
        } else {
            Some(q.to_be_bytes())
        }
    });
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn i256_mul(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 2 { polars_bail!(ComputeError: "i256_mul expects exactly 2 input columns"); }
    let s0 = &inputs[0]; let s1 = &inputs[1];
    let a = s0.binary()?; let b = s1.binary()?;
    let out = map_pair_binary_to_binary_series(s0.name(), a, b, |la, rb| {
        // Two's-complement 256-bit multiplication corresponds to wrapping mul on U256
        let ua = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(la); t });
        let ub = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(rb); t });
        Some(ua.overflowing_mul(ub).0.to_be_bytes())
    });
    Ok(out)
}

// -------------------- Series ops: cumsum/diff (u256 & i256) --------------------

#[polars_expr(output_type=Binary)]
pub fn u256_cumsum(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 { polars_bail!(ComputeError: "u256_cumsum expects 1 input column"); }
    let s0 = &inputs[0];
    let a = s0.binary()?;
    let name = s0.name();
    let mut acc = U256::from(0u8);
    let mut has_seen_valid = false;
    let iter = a.into_iter().map(|opt| {
        match opt {
            Some(bytes) if bytes.len() == 32 => {
                has_seen_valid = true;
                let v = u256_from_be32(bytes).ok()?;
                let (new, overflow) = acc.overflowing_add(v);
                acc = new;
                if overflow { None } else { Some(u256_to_be32(&acc)) }
            }
            _ => None,
        }
    });
    Ok(binary_series_from_iter(name, iter))
}

#[polars_expr(output_type=Binary)]
pub fn u256_diff(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 { polars_bail!(ComputeError: "u256_diff expects 1 input column"); }
    let s0 = &inputs[0];
    let a = s0.binary()?;
    let name = s0.name();
    let mut prev: Option<U256> = None;
    let iter = a.into_iter().map(|opt| {
        match (opt, prev) {
            (Some(bytes), None) => {
                if bytes.len() != 32 { prev = None; return None; }
                prev = u256_from_be32(bytes).ok();
                None
            }
            (Some(bytes), Some(p)) => {
                if bytes.len() != 32 { prev = None; return None; }
                let v = u256_from_be32(bytes).ok()?;
                let diff = v.checked_sub(p)?; // underflow -> None
                prev = Some(v);
                Some(u256_to_be32(&diff))
            }
            _ => { prev = None; None }
        }
    });
    Ok(binary_series_from_iter(name, iter))
}

#[polars_expr(output_type=Binary)]
pub fn i256_cumsum(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 { polars_bail!(ComputeError: "i256_cumsum expects 1 input column"); }
    let s0 = &inputs[0];
    let a = s0.binary()?;
    let name = s0.name();
    let mut acc = U256::from(0u8);
    let iter = a.into_iter().map(|opt| match opt {
        Some(bytes) if bytes.len() == 32 => {
            let v = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(bytes); t });
            let (new, _) = acc.overflowing_add(v); // wrap on overflow
            acc = new;
            Some(acc.to_be_bytes())
        }
        _ => None,
    });
    Ok(binary_series_from_iter(name, iter))
}

#[polars_expr(output_type=Binary)]
pub fn i256_diff(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 { polars_bail!(ComputeError: "i256_diff expects 1 input column"); }
    let s0 = &inputs[0];
    let a = s0.binary()?;
    let name = s0.name();
    let mut prev: Option<U256> = None;
    let iter = a.into_iter().map(|opt| {
        match (opt, prev) {
            (Some(bytes), None) => {
                if bytes.len() != 32 { prev = None; return None; }
                prev = Some(U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(bytes); t }));
                None
            }
            (Some(bytes), Some(p)) => {
                if bytes.len() != 32 { prev = None; return None; }
                let v = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(bytes); t });
                let (d, _) = v.overflowing_sub(p); // wrap on underflow
                prev = Some(v);
                Some(d.to_be_bytes())
            }
            _ => { prev = None; None }
        }
    });
    Ok(binary_series_from_iter(name, iter))
}

// -------------------- Aggregations: min/max/mean (u256 & i256) --------------------

#[polars_expr(output_type=Binary)]
pub fn u256_min(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 { polars_bail!(ComputeError: "u256_min expects exactly 1 input column"); }
    let s0 = &inputs[0];
    let a = s0.binary()?;
    let mut minv: Option<[u8;32]> = None;
    for v in a.into_iter().flatten() {
        if v.len() != 32 { continue; }
        let mut arr=[0u8;32]; arr.copy_from_slice(v);
        match &mut minv {
            None => minv = Some(arr),
            Some(m) => { if arr[..] < m[..] { *m = arr; } }
        }
    }
    let out = match minv { Some(x) => Series::new(s0.name().clone(), [Some(x.to_vec())]), None => Series::new(s0.name().clone(), [Option::<Vec<u8>>::None]) };
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn u256_max(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 { polars_bail!(ComputeError: "u256_max expects exactly 1 input column"); }
    let s0 = &inputs[0];
    let a = s0.binary()?;
    let mut maxv: Option<[u8;32]> = None;
    for v in a.into_iter().flatten() {
        if v.len() != 32 { continue; }
        let mut arr=[0u8;32]; arr.copy_from_slice(v);
        match &mut maxv {
            None => maxv = Some(arr),
            Some(m) => { if arr[..] > m[..] { *m = arr; } }
        }
    }
    let out = match maxv { Some(x) => Series::new(s0.name().clone(), [Some(x.to_vec())]), None => Series::new(s0.name().clone(), [Option::<Vec<u8>>::None]) };
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn u256_mean(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 { polars_bail!(ComputeError: "u256_mean expects exactly 1 input column"); }
    let s0 = &inputs[0];
    let a = s0.binary()?;
    let mut sum = U256::from(0u8);
    let mut cnt: u64 = 0;
    for v in a.into_iter().flatten() {
        if v.len() != 32 { continue; }
        let u = u256_from_be32(v).unwrap();
        let (new_sum, overflow) = sum.overflowing_add(u);
        if overflow { return Ok(Series::new(s0.name().clone(), [Option::<Vec<u8>>::None])); }
        sum = new_sum;
        cnt += 1;
    }
    if cnt == 0 { return Ok(Series::new(s0.name().clone(), [Option::<Vec<u8>>::None])); }
    let q = sum / U256::from(cnt);
    Ok(Series::new(s0.name().clone(), [Some(u256_to_be32(&q).to_vec())]))
}

#[polars_expr(output_type=Binary)]
pub fn i256_min(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 { polars_bail!(ComputeError: "i256_min expects exactly 1 input column"); }
    let s0 = &inputs[0];
    let a = s0.binary()?;
    let mut minv: Option<[u8;32]> = None;
    for v in a.into_iter().flatten() {
        if v.len() != 32 { continue; }
        let mut arr=[0u8;32]; arr.copy_from_slice(v);
        match &mut minv {
            None => minv = Some(arr),
            Some(m) => {
                if i256_cmp_bytes(&arr, m).map(|o| o.is_lt()).unwrap_or(false) { *m = arr; }
            }
        }
    }
    let out = match minv { Some(x) => Series::new(s0.name().clone(), [Some(x.to_vec())]), None => Series::new(s0.name().clone(), [Option::<Vec<u8>>::None]) };
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn i256_max(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 { polars_bail!(ComputeError: "i256_max expects exactly 1 input column"); }
    let s0 = &inputs[0];
    let a = s0.binary()?;
    let mut maxv: Option<[u8;32]> = None;
    for v in a.into_iter().flatten() {
        if v.len() != 32 { continue; }
        let mut arr=[0u8;32]; arr.copy_from_slice(v);
        match &mut maxv {
            None => maxv = Some(arr),
            Some(m) => {
                if i256_cmp_bytes(&arr, m).map(|o| o.is_gt()).unwrap_or(false) { *m = arr; }
            }
        }
    }
    let out = match maxv { Some(x) => Series::new(s0.name().clone(), [Some(x.to_vec())]), None => Series::new(s0.name().clone(), [Option::<Vec<u8>>::None]) };
    Ok(out)
}

#[polars_expr(output_type=Binary)]
pub fn i256_mean(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 1 { polars_bail!(ComputeError: "i256_mean expects exactly 1 input column"); }
    let s0 = &inputs[0];
    let a = s0.binary()?;
    let mut sum = U256::from(0u8);
    let mut cnt: u64 = 0;
    for v in a.into_iter().flatten() {
        if v.len() != 32 { continue; }
        let u = U256::from_be_bytes({ let mut t=[0u8;32]; t.copy_from_slice(v); t });
        // two's complement add (wrap)
        sum = sum.overflowing_add(u).0;
        cnt += 1;
    }
    if cnt == 0 { return Ok(Series::new(s0.name().clone(), [Option::<Vec<u8>>::None])); }
    // signed division of (sum) by positive cnt, trunc toward zero
    let (mag, neg) = i256_abs_u256(&sum.to_be_bytes::<32>());
    let q_abs = mag / U256::from(cnt);
    let q = if neg && q_abs != U256::from(0u8) { (!q_abs).overflowing_add(U256::from(1u8)).0 } else { q_abs };
    Ok(Series::new(s0.name().clone(), [Some(q.to_be_bytes::<32>().to_vec())]))
}
