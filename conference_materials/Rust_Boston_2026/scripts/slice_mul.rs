fn mul(a: &[f64], b: &[f64]) -> Result<Vec<f64>, &'static str> {
    // Check bounds before loop!
    // Otherwise, it will not vectorize
    let n: usize = a.len();
    if b.len() != n {
        return Err("Dimension mismatch");
    }

    // Allocate storage
    let mut out: Vec<f64> = vec![0.0; n];

    // Do the calculations
    for i: usize in 0..n {
        out[i] = a[i] * b[i];
    }

    Ok(out)
}
