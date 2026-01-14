#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! libm = "0.2.15"
//! rayon = "1"
//! num_cpus = "1"
//! ```

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot_fma(u: [f64; 3], v: [f64; 3]) -> f64 {
    u[0].mul_add(v[0], u[1].mul_add(v[1], u[2] * v[2]))
}

#[inline]
fn cross_fma(u: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        u[1].mul_add(v[2], - u[2] * v[1]),
        u[2].mul_add(v[0], - u[0] * v[2]),
        u[0].mul_add(v[1], - u[1] * v[0]),
    ]
}

#[inline]
fn norm(u: [f64; 3]) -> f64 {
    dot_fma(u, u).sqrt()
}

/// Angular portion of a sphere subtended by a
/// tetrahedron with the first vertex as the origin.
/// https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron
#[inline]
pub fn solid_angle_tetrahedron_scalar_fma(
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    v3: [f64; 3],
) -> f64 {
    // Vertex vectors
    let (a, b, c) = (sub(v1, v0), sub(v2, v0), sub(v3, v0));  // (m)
    let (la, lb, lc) = (norm(a), norm(b), norm(c)); // (m) Vertex vector lengths
    let abc = la * lb * lc;  // (m^3) Length product. Branch determined here!

    // Solid angle
    let triple = dot_fma(a, cross_fma(b, c)); // (m^3) Scalar triple product
    let denom = dot_fma(a, b).mul_add(lc, dot_fma(a, c).mul_add(lb, dot_fma(b, c).mul_add(la, abc)));  // (m^3)
    // let angle = 2.0 * libm::atan2(triple, denom); // (rad) f64::atan2 defers to libc
    let angle = 2.0 * triple.atan2(denom);

    // Check for degeneracy _last_ to avoid disrupting flow
    if abc != 0.0 {
        angle
    } else {
        0.0
    }
}

/// Vector variant of [solid_angle_tetrahedron_scalar]
#[inline] // Enable cross-crate inlining
#[unsafe(no_mangle)]
pub fn solid_angle_tetrahedron(
    tetrahedra: &[[[f64; 3]; 4]],
    out: &mut [f64],
) -> Result<(), &'static str> {
    // Check bounds
    let n = out.len();
    if tetrahedra.len() != n {
        return Err("Dimension mismatch");
    }

    // Do calculations
    for i in 0..n {
        let tet = tetrahedra[i];
        out[i] = solid_angle_tetrahedron_scalar_fma(tet[0], tet[1], tet[2], tet[3]);
    }

    Ok(())
}
