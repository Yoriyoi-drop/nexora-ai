//! Tensor Pool untuk mengurangi alokasi berulang
//!
//! Implementasi object pooling untuk tensor yang sering digunakan:
//! - Reusable tensor buffers
//! - Memory pre-allocation
//! - Automatic garbage collection
//! - Thread-safe operations

use crate::{DLResult, DeepLearningError};
use ndarray::{ArrayD, Array1, Array2, s};
use std::sync::{Arc, Mutex};

/// Tensor pool untuk berbagai ukuran tensor
#[derive(Debug, Clone)]
pub struct TensorPool {
    // Pool untuk 1D arrays
    pool_1d: Arc<Mutex<Vec<Vec<Array1<f32>>>>>,
    // Pool untuk 2D arrays  
    pool_2d: Arc<Mutex<Vec<Vec<Array2<f32>>>>>,
    // Pool untuk dynamic arrays
    pool_dyn: Arc<Mutex<Vec<Vec<ArrayD<f32>>>>>,
    
    // Size categories untuk setiap pool
    size_categories_1d: Vec<usize>,
    size_categories_2d: Vec<(usize, usize)>,
    size_categories_dyn: Vec<Vec<usize>>,
    
    // Maximum pool sizes
    max_pool_size: usize,
}

impl TensorPool {
    /// Create new tensor pool dengan predefined size categories
    pub fn new() -> DLResult<Self> {
        let size_categories_1d = vec![64, 128, 256, 512, 1024, 2048];
        let size_categories_2d = vec![
            (64, 64), (128, 128), (256, 256), (512, 512),
            (64, 512), (128, 1024), (256, 2048)
        ];
        let size_categories_dyn = vec![
            vec![512], vec![1024], vec![2048],
            vec![64, 64], vec![128, 128], vec![256, 256]
        ];
        
        let pool_1d = Arc::new(Mutex::new(vec![Vec::new(); size_categories_1d.len()]));
        let pool_2d = Arc::new(Mutex::new(vec![Vec::new(); size_categories_2d.len()]));
        let pool_dyn = Arc::new(Mutex::new(vec![Vec::new(); size_categories_dyn.len()]));
        
        Ok(Self {
            pool_1d,
            pool_2d,
            pool_dyn,
            size_categories_1d,
            size_categories_2d,
            size_categories_dyn,
            max_pool_size: 100,
        })
    }
    
    /// Get 1D tensor from pool atau create new
    pub fn get_1d(&self, size: usize) -> DLResult<Array1<f32>> {
        let mut pool = self.pool_1d.lock().unwrap();
        
        // Find appropriate size category
        let category_idx = self.find_1d_category(size)?;
        
        if let Some(tensor) = pool[category_idx].pop() {
            if tensor.len() >= size {
                return Ok(tensor.slice(s![0..size]).to_owned());
            }
        }
        
        // Create new tensor if pool is empty or too small
        Ok(Array1::zeros(size))
    }
    
    /// Return 1D tensor to pool
    pub fn return_1d(&self, tensor: Array1<f32>) -> DLResult<()> {
        let size = tensor.len();
        let category_idx = self.find_1d_category(size)?;
        
        let mut pool = self.pool_1d.lock().unwrap();
        if pool[category_idx].len() < self.max_pool_size {
            pool[category_idx].push(tensor);
        }
        
        Ok(())
    }
    
    /// Get 2D tensor from pool atau create new
    pub fn get_2d(&self, rows: usize, cols: usize) -> DLResult<Array2<f32>> {
        let mut pool = self.pool_2d.lock().unwrap();
        
        // Find appropriate size category
        let category_idx = self.find_2d_category(rows, cols)?;
        
        if let Some(tensor) = pool[category_idx].pop() {
            if tensor.nrows() >= rows && tensor.ncols() >= cols {
                return Ok(tensor.slice(s![0..rows, 0..cols]).to_owned());
            }
        }
        
        // Create new tensor if pool is empty or too small
        Ok(Array2::zeros((rows, cols)))
    }
    
    /// Return 2D tensor to pool
    pub fn return_2d(&self, tensor: Array2<f32>) -> DLResult<()> {
        let (rows, cols) = tensor.dim();
        let category_idx = self.find_2d_category(rows, cols)?;
        
        let mut pool = self.pool_2d.lock().unwrap();
        if pool[category_idx].len() < self.max_pool_size {
            pool[category_idx].push(tensor);
        }
        
        Ok(())
    }
    
    /// Get dynamic tensor from pool atau create new
    pub fn get_dyn(&self, shape: &[usize]) -> DLResult<ArrayD<f32>> {
        let mut pool = self.pool_dyn.lock().unwrap();
        
        // Find appropriate size category
        let category_idx = self.find_dyn_category(shape)?;
        
        if let Some(tensor) = pool[category_idx].pop() {
            if tensor.shape().iter().zip(shape.iter()).all(|(&t, &s)| t >= s) {
                // For now, just return the tensor as-is (slicing would need more complex implementation)
                return Ok(tensor.to_owned());
            }
        }
        
        // Create new tensor if pool is empty or too small
        Ok(ArrayD::zeros(shape))
    }
    
    /// Return dynamic tensor to pool
    pub fn return_dyn(&self, tensor: ArrayD<f32>) -> DLResult<()> {
        let shape = tensor.shape().to_vec();
        let category_idx = self.find_dyn_category(&shape)?;
        
        let mut pool = self.pool_dyn.lock().unwrap();
        if pool[category_idx].len() < self.max_pool_size {
            pool[category_idx].push(tensor);
        }
        
        Ok(())
    }
    
    /// Find appropriate 1D size category
    fn find_1d_category(&self, size: usize) -> DLResult<usize> {
        self.size_categories_1d
            .iter()
            .position(|&cat_size| cat_size >= size)
            .ok_or_else(|| DeepLearningError::Configuration {
                reason: format!("No size category found for 1D tensor of size {}", size),
            })
    }
    
    /// Find appropriate 2D size category
    fn find_2d_category(&self, rows: usize, cols: usize) -> DLResult<usize> {
        self.size_categories_2d
            .iter()
            .position(|&(cat_rows, cat_cols)| cat_rows >= rows && cat_cols >= cols)
            .ok_or_else(|| DeepLearningError::Configuration {
                reason: format!("No size category found for 2D tensor of shape ({}, {})", rows, cols),
            })
    }
    
    /// Find appropriate dynamic size category
    fn find_dyn_category(&self, shape: &[usize]) -> DLResult<usize> {
        self.size_categories_dyn
            .iter()
            .position(|cat_shape| {
                cat_shape.len() == shape.len() && 
                cat_shape.iter().zip(shape.iter()).all(|(&cat, &req)| cat >= req)
            })
            .ok_or_else(|| DeepLearningError::Configuration {
                reason: format!("No size category found for dynamic tensor of shape {:?}", shape),
            })
    }
    
    /// Clear all pools (for memory cleanup)
    pub fn clear_all(&self) -> DLResult<()> {
        {
            let mut pool = self.pool_1d.lock().unwrap();
            for category in pool.iter_mut() {
                category.clear();
            }
        }
        {
            let mut pool = self.pool_2d.lock().unwrap();
            for category in pool.iter_mut() {
                category.clear();
            }
        }
        {
            let mut pool = self.pool_dyn.lock().unwrap();
            for category in pool.iter_mut() {
                category.clear();
            }
        }
        Ok(())
    }
    
    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStats {
        let mut stats = PoolStats::default();
        
        {
            let pool = self.pool_1d.lock().unwrap();
            for (i, category) in pool.iter().enumerate() {
                stats.pool_1d_sizes.push((self.size_categories_1d[i], category.len()));
            }
        }
        
        {
            let pool = self.pool_2d.lock().unwrap();
            for (i, category) in pool.iter().enumerate() {
                stats.pool_2d_sizes.push((self.size_categories_2d[i], category.len()));
            }
        }
        
        {
            let pool = self.pool_dyn.lock().unwrap();
            for (i, category) in pool.iter().enumerate() {
                stats.pool_dyn_sizes.push((self.size_categories_dyn[i].clone(), category.len()));
            }
        }
        
        stats
    }
}

/// Pool statistics
#[derive(Debug, Default, Clone)]
pub struct PoolStats {
    pub pool_1d_sizes: Vec<(usize, usize)>,
    pub pool_2d_sizes: Vec<((usize, usize), usize)>,
    pub pool_dyn_sizes: Vec<(Vec<usize>, usize)>,
}

/// Global tensor pool instance
static GLOBAL_TENSOR_POOL: std::sync::LazyLock<TensorPool> = std::sync::LazyLock::new(|| {
    TensorPool::new().expect("Failed to create global tensor pool")
});

/// Get global tensor pool
pub fn global_pool() -> &'static TensorPool {
    &GLOBAL_TENSOR_POOL
}

/// RAII wrapper untuk automatic tensor return
pub struct PooledTensor1D {
    tensor: Array1<f32>,
    returned: bool,
}

impl PooledTensor1D {
    pub fn new(size: usize) -> DLResult<Self> {
        let tensor = global_pool().get_1d(size)?;
        Ok(Self {
            tensor,
            returned: false,
        })
    }
    
    pub fn get(&self) -> &Array1<f32> {
        &self.tensor
    }
    
    pub fn get_mut(&mut self) -> &mut Array1<f32> {
        &mut self.tensor
    }
    
    pub fn into_inner(mut self) -> Array1<f32> {
        self.returned = true;
        self.tensor.clone()
    }
}

impl Drop for PooledTensor1D {
    fn drop(&mut self) {
        if !self.returned {
            let _ = global_pool().return_1d(self.tensor.clone());
        }
    }
}

/// RAII wrapper untuk 2D tensors
pub struct PooledTensor2D {
    tensor: Array2<f32>,
    returned: bool,
}

impl PooledTensor2D {
    pub fn new(rows: usize, cols: usize) -> DLResult<Self> {
        let tensor = global_pool().get_2d(rows, cols)?;
        Ok(Self {
            tensor,
            returned: false,
        })
    }
    
    pub fn get(&self) -> &Array2<f32> {
        &self.tensor
    }
    
    pub fn get_mut(&mut self) -> &mut Array2<f32> {
        &mut self.tensor
    }
    
    pub fn into_inner(mut self) -> Array2<f32> {
        self.returned = true;
        self.tensor.clone()
    }
}

impl Drop for PooledTensor2D {
    fn drop(&mut self) {
        if !self.returned {
            let _ = global_pool().return_2d(self.tensor.clone());
        }
    }
}

/// RAII wrapper untuk dynamic tensors
pub struct PooledTensorDyn {
    tensor: ArrayD<f32>,
    returned: bool,
}

impl PooledTensorDyn {
    pub fn new(shape: &[usize]) -> DLResult<Self> {
        let tensor = global_pool().get_dyn(shape)?;
        Ok(Self {
            tensor,
            returned: false,
        })
    }
    
    pub fn get(&self) -> &ArrayD<f32> {
        &self.tensor
    }
    
    pub fn get_mut(&mut self) -> &mut ArrayD<f32> {
        &mut self.tensor
    }
    
    pub fn into_inner(mut self) -> ArrayD<f32> {
        self.returned = true;
        self.tensor.clone()
    }
}

impl Drop for PooledTensorDyn {
    fn drop(&mut self) {
        if !self.returned {
            let _ = global_pool().return_dyn(self.tensor.clone());
        }
    }
}
