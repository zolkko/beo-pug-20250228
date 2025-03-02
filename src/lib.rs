use pyo3::prelude::*;
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use ndarray::prelude::*;

type Double = f64;

fn rolling_mean(l: &ArrayView1<Double>, k: usize, res: &mut ArrayViewMut1<Double>) {
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
fn rolling_mean_v9<'py>(py: Python<'py>, l: Bound<'py, PyArray1<Double>>, k: usize) -> PyResult<Bound<'py, PyArray1<Double>>> {
    let l_array_ro = l.try_readonly()?;
    let l_array = l_array_ro.as_array();

    // allocate using numpy heap
    let res = PyArray1::<Double>::zeros(py, l.shape()[0], true);
    // SAFETY: we have just created the array and the array is contiguous and has one dimension.
    let mut nd_res = unsafe { res.as_array_mut() };

    py.allow_threads(|| {
        rolling_mean(&l_array, k, &mut nd_res);
    });

    Ok(res)
}

#[pymodule]
fn rst_wnd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rolling_mean_v9, m)?)?;
    Ok(())
}
