#![feature(portable_simd)]

use std::mem::MaybeUninit;
use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use ndarray::prelude::*;

type Double = f64;

fn rolling_mean_safe(l: &ArrayView1<Double>, k: usize, res: &mut ArrayViewMut1<Double>) {
    let n = l.shape()[0];
    let k_inv = (k as Double).recip();

    res.slice_mut(s![..k-1]).fill(Double::NAN);
    let mut s = l.slice(s![..k]).sum();

    res[k-1] = s * k_inv;

    for i in k..n {
        s += l[i] - l[i-k];
        res[i] = s * k_inv;
    }
}

#[pyfunction]
fn rolling_mean_v9_safe<'py>(py: Python<'py>, l: Bound<'py, PyArray1<Double>>, k: usize) -> PyResult<Bound<'py, PyArray1<Double>>> {
    let l_array_ro = l.try_readonly()?;
    let l_array = l_array_ro.as_array();

    // allocate using numpy heap
    let res = PyArray1::<Double>::zeros(py, l.shape()[0], true);
    // SAFETY: we have just created the array and the array is contiguous and has one dimension.
    let mut nd_res = unsafe { res.as_array_mut() };

    py.allow_threads(|| {
        rolling_mean_safe(&l_array, k, &mut nd_res);
    });

    Ok(res)
}

fn rolling_mean_raw_pointers(l: &ArrayView1<Double>, k: usize, res: &mut ArrayViewMut1<Double>) {
    let res_ptr = res.as_mut_ptr();

    for i in 0..k-1 {
        unsafe {
            *res_ptr.add(i) = Double::NAN;
        }
    }

    let mut s = 0.0;
    let mut l_ptr = l.as_ptr();
    for i in 0..k {
        unsafe {
            s += *l_ptr;
            l_ptr = l_ptr.add(i);
        }
    }

    let k_inv = (k as Double).recip();
    unsafe {
        *res_ptr.add(k - 1) = s * k_inv;
    }

    let n = l.shape()[0];
    for i in k..n {
        unsafe {
            s += *l_ptr.add(i) - *l_ptr.add(i - k);
            *res_ptr.add(i) = s * k_inv;
        }
    }
}

#[pyfunction]
fn rolling_mean_v9_raw_pointers<'py>(py: Python<'py>, l: Bound<'py, PyArray1<Double>>, k: usize) -> PyResult<Bound<'py, PyArray1<Double>>> {
    let l_array_ro = l.try_readonly()?;
    let l_array = l_array_ro.as_array();

    // allocate using numpy heap
    let res = PyArray1::<Double>::zeros(py, l.shape()[0], true);
    let mut res_rw = res.readwrite();
    let mut res_mut_view = res_rw.as_array_mut();

    py.allow_threads(|| {
        rolling_mean_raw_pointers(&l_array, k, &mut res_mut_view);
    });

    Ok(res)
}

fn rolling_mean_portable_simd(l: &ArrayView1<Double>, k: usize, res: &mut ArrayViewMut1<Double>) {
    use std::simd::prelude::*;

    const N: usize = 4;

    let res_slice = res.as_slice_mut().expect("res is not contiguous");
    {
        let nan_vec = Simd::<Double, N>::splat(Double::NAN);
        let aligned_end = (k - 1) / N * N;

        for i in (0..aligned_end).step_by(N) {
            nan_vec.copy_to_slice(&mut res_slice[i..(i + N)]);
        }

        for val in &mut res_slice[aligned_end..(k - 1)] {
            *val = Double::NAN;
        }
    };

    let l_slice = l.as_slice().expect("l is not contiguous");
    let mut s = {
        let aligned_end = k / N * N;
        let mut s_es = Simd::<Double, N>::splat(0.0);

        for i in (0..aligned_end).step_by(N) {
            let data = Simd::<Double, N>::from_slice(&l_slice[i..i + N]);
            s_es += data;
        }

        let mut s: Double = s_es.to_array().iter().sum();
        for val in &res_slice[aligned_end..k] {
            s += *val;
        }

        s
    };

    let k_inv = (k as Double).recip();

    res_slice[k - 1] = s * k_inv;

    for i in k..l_slice.len() {
        s += l_slice[i] - l_slice[i - k];
        res_slice[i] = s * k_inv;
    }
}

#[pyfunction]
fn rolling_mean_v9_portable_simd<'py>(py: Python<'py>, l: Bound<'py, PyArray1<Double>>, k: usize) -> PyResult<Bound<'py, PyArray1<Double>>> {
    let l_array_ro = l.try_readonly()?;
    let l_array = l_array_ro.as_array();

    // allocate using numpy heap
    let res = PyArray1::<Double>::zeros(py, l.shape()[0], true);
    // SAFETY: we have just created the array and the array is contiguous and has one dimension.
    let mut nd_res = unsafe { res.as_array_mut() };

    py.allow_threads(|| {
        rolling_mean_portable_simd(&l_array, k, &mut nd_res);
    });

    Ok(res)
}

fn rolling_mean_simd_intr(l: &ArrayView1<Double>, k: usize, res: &mut ArrayViewMut1<Double>) {
    use std::arch::x86_64::{_mm256_add_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_loadu_pd, _mm256_hadd_pd};

    let n = l.shape()[0];
    let k_inv = (k as Double).recip();

    let res_ptr = res.as_mut_ptr();
    unsafe {
        let aligned_end = (k - 1) / 4 * 4;
        let nan_vec = _mm256_set1_pd(Double::NAN);
        for i in (0..aligned_end).step_by(4) {
            _mm256_storeu_pd(res_ptr.add(i), nan_vec);
        }

        for i in aligned_end..(k - 1) {
            *res_ptr.add(i) = Double::NAN;
        }
    }

    let l_ptr = l.as_ptr();
    let mut s = unsafe {
        let mut s_es = _mm256_set1_pd(0.0);
        let aligned_end = k / 4 * 4;
        for i in (0..aligned_end).step_by(4) {
            let data = _mm256_loadu_pd(l_ptr.add(i));
            s_es = _mm256_add_pd(s_es, data);
        }
        s_es = _mm256_hadd_pd(s_es, s_es);

        let s_slice: [Double; 4] = std::mem::transmute(s_es);
        let mut s = s_slice[0] + s_slice[1] + s_slice[2] + s_slice[3];

        for i in aligned_end..k {
            s += *l_ptr.add(i);
        }

        s
    };

    unsafe {
        *res_ptr.add(k - 1) = s * k_inv;
    }

    for i in k..n {
        unsafe {
            s += *l_ptr.add(i) - *l_ptr.add(i - k);
            *res_ptr.add(i) = s * k_inv;
        }
    }
}

#[pyfunction]
fn rolling_mean_v9_simd_intr<'py>(py: Python<'py>, l: Bound<'py, PyArray1<Double>>, k: usize) -> PyResult<Bound<'py, PyArray1<Double>>> {
    let l_array_ro = l.try_readonly()?;
    let l_array = l_array_ro.as_array();

    // allocate using numpy heap
    let res = PyArray1::<Double>::zeros(py, l.shape()[0], true);
    // SAFETY: we have just created the array and the array is contiguous and has one dimension.
    let mut nd_res = unsafe { res.as_array_mut() };

    py.allow_threads(|| {
        rolling_mean_simd_intr(&l_array, k, &mut nd_res);
    });

    Ok(res)
}

#[pymodule]
fn rst_wnd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rolling_mean_v9_safe, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_v9_raw_pointers, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_v9_portable_simd, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_v9_simd_intr, m)?)?;
    Ok(())
}
