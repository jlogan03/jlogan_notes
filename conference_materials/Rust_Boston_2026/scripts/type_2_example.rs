#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! libm = "0.2.15"
//! rayon = "1"
//! ```

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(u: [f64; 3], v: [f64; 3]) -> f64 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}

#[inline]
fn cross(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

#[inline]
fn norm(u: [f64; 3]) -> f64 {
    dot(u, u).sqrt()
}

/// Angular portion of a sphere subtended by a
/// tetrahedron with the first vertex as the origin.
/// https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron
#[inline]
pub fn solid_angle_tetrahedron_scalar(
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    v3: [f64; 3],
) -> f64 {
    // Vertex vectors
    let (a, b, c) = (sub(v1, v0), sub(v2, v0), sub(v3, v0));
    let (la, lb, lc) = (norm(a), norm(b), norm(c)); // Vertex vector lengths

    // Check for degeneracy
    if la == 0.0 || lb == 0.0 || lc == 0.0 {
        return 0.0;
    }

    let triple = dot(a, cross(b, c)); // Scalar triple product
    let denom = la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la;

    2.0 * libm::atan2(triple, denom) // f64::atan2 defers to libc
}

/// Vector variant of [solid_angle_tetrahedron_scalar]
#[inline] // Enable cross-crate inlining
pub fn solid_angle_tetrahedron(
    tetrahedra: &[[[f64; 3]; 4]],
    out: &mut [f64],
) -> Result<(), &'static str> {
    // Check bounds
    if out.len() != tetrahedra.len() {
        return Err("Dimension mismatch");
    }

    // Do calculations
    for (i, tet) in tetrahedra.iter().enumerate() {
        out[i] = solid_angle_tetrahedron_scalar(tet[0], tet[1], tet[2], tet[3]);
    }

    Ok(())
}

use rayon::prelude::*;

/// Vector-parallel variant of [solid_angle_tetrahedron_scalar]
pub fn solid_angle_tetrahedra_par(
    tetrahedra: &[[[f64; 3]; 4]],
    out: &mut [f64],
) -> Result<(), &'static str> {
    // Chunk inputs
    let num_chunks = 1024.min(out.len() / rayon::current_num_threads());
    let (out_chunks, tet_chunks) = (
        out.par_chunks_mut(num_chunks),
        tetrahedra.par_chunks(num_chunks),
    );

    // Do vector calculations over each chunk in parallel
    (out_chunks, tet_chunks)
        .into_par_iter()
        .try_for_each(|(outc, tetc)| solid_angle_tetrahedron(tetc, outc))?;

    Ok(())
}
