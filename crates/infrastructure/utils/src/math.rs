//! Mathematical utilities untuk Nexora


pub struct MathUtils;

impl MathUtils {
    /// Calculate mean of numbers
    pub fn mean(numbers: &[f64]) -> Option<f64> {
        if numbers.is_empty() {
            None
        } else {
            let sum: f64 = numbers.iter().sum();
            Some(sum / numbers.len() as f64)
        }
    }
    
    /// Calculate median of numbers
    pub fn median(numbers: &mut [f64]) -> Option<f64> {
        if numbers.is_empty() {
            None
        } else {
            numbers.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let len = numbers.len();
            if len % 2 == 0 {
                Some((numbers[len / 2 - 1] + numbers[len / 2]) / 2.0)
            } else {
                Some(numbers[len / 2])
            }
        }
    }
    
    /// Calculate mode of numbers
    pub fn mode(numbers: &[f64]) -> Option<Vec<f64>> {
        if numbers.is_empty() {
            return None;
        }
        
        // Use a simple approach for mode calculation since f64 doesn't implement Ord
        let mut frequency: Vec<(f64, usize)> = Vec::new();
        for &num in numbers {
            if let Some(entry) = frequency.iter_mut().find(|(val, _)| *val == num) {
                entry.1 += 1;
            } else {
                frequency.push((num, 1));
            }
        }
        
        let max_freq = frequency.iter().map(|(_, freq)| *freq).max().unwrap();
        let modes: Vec<f64> = frequency
            .into_iter()
            .filter(|(_, freq)| *freq == max_freq)
            .map(|(num, _)| num)
            .collect();
        
        Some(modes)
    }
    
    /// Calculate standard deviation
    pub fn std_deviation(numbers: &[f64]) -> Option<f64> {
        if numbers.is_empty() {
            None
        } else {
            let mean = Self::mean(numbers)?;
            let variance = numbers
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / numbers.len() as f64;
            Some(variance.sqrt())
        }
    }
    
    /// Calculate variance
    pub fn variance(numbers: &[f64]) -> Option<f64> {
        if numbers.is_empty() {
            None
        } else {
            let mean = Self::mean(numbers)?;
            Some(numbers
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / numbers.len() as f64)
        }
    }
    
    /// Calculate percentile
    pub fn percentile(numbers: &mut [f64], percentile: f64) -> Option<f64> {
        if numbers.is_empty() || percentile < 0.0 || percentile > 100.0 {
            None
        } else {
            numbers.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let index = (percentile / 100.0 * (numbers.len() - 1) as f64) as usize;
            Some(numbers[index])
        }
    }
    
    /// Calculate correlation coefficient
    pub fn correlation(x: &[f64], y: &[f64]) -> Option<f64> {
        if x.len() != y.len() || x.is_empty() {
            None
        } else {
            let x_mean = Self::mean(x)?;
            let y_mean = Self::mean(y)?;
            
            let numerator: f64 = x.iter()
                .zip(y.iter())
                .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
                .sum();
            
            let x_std = Self::std_deviation(x)?;
            let y_std = Self::std_deviation(y)?;
            
            let denominator = x_std * y_std * x.len() as f64;
            
            if denominator == 0.0 {
                None
            } else {
                Some(numerator / denominator)
            }
        }
    }
    
    /// Round number to specified decimal places
    pub fn round(number: f64, decimal_places: usize) -> f64 {
        let multiplier = 10_f64.powi(decimal_places as i32);
        (number * multiplier).round() / multiplier
    }
    
    /// Clamp number between min and max
    pub fn clamp(number: f64, min: f64, max: f64) -> f64 {
        number.max(min).min(max)
    }
    
    /// Linear interpolation
    pub fn lerp(start: f64, end: f64, t: f64) -> f64 {
        start + (end - start) * t.clamp(0.0, 1.0)
    }
    
    /// Normalize numbers to [0, 1] range
    pub fn normalize(numbers: &[f64]) -> Option<Vec<f64>> {
        if numbers.is_empty() {
            None
        } else {
            let min = numbers.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = numbers.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            if max == min {
                Some(vec![0.5; numbers.len()])
            } else {
                Some(numbers.iter().map(|x| (x - min) / (max - min)).collect())
            }
        }
    }
    
    /// Calculate Euclidean distance between two vectors
    pub fn euclidean_distance(a: &[f64], b: &[f64]) -> Option<f64> {
        if a.len() != b.len() {
            None
        } else {
            Some(a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt())
        }
    }
    
    /// Calculate Manhattan distance between two vectors
    pub fn manhattan_distance(a: &[f64], b: &[f64]) -> Option<f64> {
        if a.len() != b.len() {
            None
        } else {
            Some(a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .sum())
        }
    }
    
    /// Calculate cosine similarity between two vectors
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Option<f64> {
        if a.len() != b.len() || a.is_empty() {
            None
        } else {
            let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let magnitude_a: f64 = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            let magnitude_b: f64 = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
            
            if magnitude_a == 0.0 || magnitude_b == 0.0 {
                None
            } else {
                Some(dot_product / (magnitude_a * magnitude_b))
            }
        }
    }
    
    /// Generate random number in range [min, max]
    pub fn random_range(min: f64, max: f64) -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(min..=max)
    }
    
    /// Generate random integer in range [min, max]
    pub fn random_int_range(min: i32, max: i32) -> i32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(min..=max)
    }
    
    /// Check if number is prime
    pub fn is_prime(n: u64) -> bool {
        if n <= 1 {
            false
        } else if n <= 3 {
            true
        } else if n % 2 == 0 || n % 3 == 0 {
            false
        } else {
            let mut i = 5;
            while i * i <= n {
                if n % i == 0 || n % (i + 2) == 0 {
                    return false;
                }
                i += 6;
            }
            true
        }
    }
    
    /// Calculate factorial
    pub fn factorial(n: u64) -> Option<u64> {
        if n > 20 {
            None // Would overflow u64
        } else {
            Some((1..=n).product())
        }
    }
    
    /// Calculate greatest common divisor
    pub fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            Self::gcd(b, a % b)
        }
    }
    
    /// Calculate least common multiple
    pub fn lcm(a: u64, b: u64) -> u64 {
        a * b / Self::gcd(a, b)
    }
    
    /// Check if number is even
    pub fn is_even(n: u64) -> bool {
        n % 2 == 0
    }
    
    /// Check if number is odd
    pub fn is_odd(n: u64) -> bool {
        n % 2 != 0
    }
    
    /// Convert degrees to radians
    pub fn degrees_to_radians(degrees: f64) -> f64 {
        degrees * std::f64::consts::PI / 180.0
    }
    
    /// Convert radians to degrees
    pub fn radians_to_degrees(radians: f64) -> f64 {
        radians * 180.0 / std::f64::consts::PI
    }
    
    /// Calculate sigmoid function
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Calculate ReLU function
    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }
    
    /// Calculate tanh function
    pub fn tanh(x: f64) -> f64 {
        x.tanh()
    }
    
    /// Calculate softmax for vector
    pub fn softmax(numbers: &[f64]) -> Option<Vec<f64>> {
        if numbers.is_empty() {
            None
        } else {
            let max_val = numbers.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_values: Vec<f64> = numbers.iter()
                .map(|x| (x - max_val).exp())
                .collect();
            let sum_exp: f64 = exp_values.iter().sum();
            
            if sum_exp == 0.0 {
                None
            } else {
                Some(exp_values.iter().map(|x| x / sum_exp).collect())
            }
        }
    }
    
    /// Calculate moving average
    pub fn moving_average(numbers: &[f64], window_size: usize) -> Option<Vec<f64>> {
        if numbers.is_empty() || window_size == 0 || window_size > numbers.len() {
            None
        } else {
            let mut result = Vec::new();
            for i in 0..=numbers.len() - window_size {
                let window = &numbers[i..i + window_size];
                result.push(Self::mean(window)?);
            }
            Some(result)
        }
    }
    
    /// Calculate exponential moving average
    pub fn exponential_moving_average(numbers: &[f64], alpha: f64) -> Option<Vec<f64>> {
        if numbers.is_empty() || alpha <= 0.0 || alpha > 1.0 {
            None
        } else {
            let mut result = Vec::new();
            let mut ema = numbers[0];
            result.push(ema);
            
            for &value in &numbers[1..] {
                ema = alpha * value + (1.0 - alpha) * ema;
                result.push(ema);
            }
            
            Some(result)
        }
    }
    
    /// Find outliers using IQR method
    pub fn find_outliers_iqr(numbers: &[f64]) -> Option<Vec<f64>> {
        if numbers.len() < 4 {
            None
        } else {
            let mut sorted = numbers.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let q1 = Self::percentile(&mut sorted, 25.0)?;
            let q3 = Self::percentile(&mut sorted, 75.0)?;
            let iqr = q3 - q1;
            let lower_bound = q1 - 1.5 * iqr;
            let upper_bound = q3 + 1.5 * iqr;
            
            let outliers: Vec<f64> = numbers.iter()
                .filter(|&&x| x < lower_bound || x > upper_bound)
                .copied()
                .collect();
            
            Some(outliers)
        }
    }
    
    /// Calculate z-score for each value
    pub fn z_scores(numbers: &[f64]) -> Option<Vec<f64>> {
        if numbers.is_empty() {
            None
        } else {
            let mean = Self::mean(numbers)?;
            let std_dev = Self::std_deviation(numbers)?;
            
            if std_dev == 0.0 {
                None
            } else {
                Some(numbers.iter().map(|x| (x - mean) / std_dev).collect())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_math_utils() {
        let numbers = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Test mean
        assert_eq!(MathUtils::mean(&numbers), Some(3.0));
        
        // Test median
        let mut numbers_copy = numbers.clone();
        assert_eq!(MathUtils::median(&mut numbers_copy), Some(3.0));
        
        // Test standard deviation
        assert!(MathUtils::std_deviation(&numbers).unwrap() > 0.0);
        
        // Test correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = MathUtils::correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 0.001);
        
        // Test Euclidean distance
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 8.0];
        let dist = MathUtils::euclidean_distance(&a, &b).unwrap();
        assert!((dist - 7.071).abs() < 0.01);
        
        // Test prime
        assert!(MathUtils::is_prime(7));
        assert!(!MathUtils::is_prime(8));
        
        // Test factorial
        assert_eq!(MathUtils::factorial(5), Some(120));
        
        // Test softmax
        let softmax = MathUtils::softmax(&[1.0, 2.0, 3.0]).unwrap();
        assert!((softmax.iter().sum::<f64>() - 1.0).abs() < 0.001);
    }
}
