//! Self-Play with Instruction Following (SPIN) Implementation
//! 
//! SPIN memungkinkan model berlatih tanding melawan dirinya sendiri
//! untuk menjadi lebih kuat secara iteratif.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use rand::seq::SliceRandom;

use super::core::{PolicyModel, ReasoningTrace, JudgeFeedback, FeedbackType};

/// Konfigurasi SPIN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpinConfig {
    /// Number of self-play rounds per iteration
    pub rounds_per_iteration: usize,
    /// Temperature untuk sampling
    pub temperature: f32,
    /// Top-p sampling threshold
    pub top_p: f32,
    /// Minimum improvement threshold
    pub improvement_threshold: f32,
    /// Maximum attempts per prompt
    pub max_attempts: usize,
    /// Maximum steps per reasoning trace
    pub max_steps: usize,
}

impl Default for SpinConfig {
    fn default() -> Self {
        Self {
            rounds_per_iteration: 3,
            temperature: 0.7,
            top_p: 0.9,
            improvement_threshold: 0.05,
            max_attempts: 5,
            max_steps: 10,
        }
    }
}

/// Self-play game state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfPlayGame {
    pub id: Uuid,
    pub prompt: String,
    pub student_trace: ReasoningTrace,
    pub teacher_trace: ReasoningTrace,
    pub round_number: usize,
    pub student_score: f32,
    pub teacher_score: f32,
    pub winner: GameWinner,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Winner of self-play game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameWinner {
    Student,
    Teacher,
    Draw,
}

/// Self-play tournament
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfPlayTournament {
    pub id: Uuid,
    pub games: Vec<SelfPlayGame>,
    pub student_wins: usize,
    pub teacher_wins: usize,
    pub draws: usize,
    pub student_win_rate: f32,
    pub improvement_score: f32,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// SPIN Generator untuk menghasilkan reasoning traces
pub struct SpinGenerator {
    config: SpinConfig,
}

impl SpinGenerator {
    pub fn new(config: SpinConfig) -> Self {
        Self { config }
    }
    
    /// Generate reasoning trace dari model
    pub fn generate_trace(&self, model: &PolicyModel, prompt: &str) -> Result<ReasoningTrace> {
        let steps = self.generate_reasoning_steps(model, prompt)?;
        let final_answer = self.generate_final_answer(&steps);
        
        Ok(ReasoningTrace {
            id: Uuid::new_v4(),
            prompt: prompt.to_string(),
            steps,
            final_answer,
            created_at: chrono::Utc::now(),
        })
    }
    
    /// Generate reasoning steps
    fn generate_reasoning_steps(&self, model: &PolicyModel, prompt: &str) -> Result<Vec<super::core::ReasoningStep>> {
        let step_count = (prompt.len() / 50).max(1).min(self.config.max_steps);
        let mut steps = Vec::with_capacity(step_count);
        
        for i in 1..=step_count {
            let step_content = format!("Step {}: Reasoning about {}", i, prompt);
            
            let step = super::core::ReasoningStep {
                id: Uuid::new_v4(),
                content: step_content,
                step_number: i,
                timestamp: chrono::Utc::now(),
            };
            
            steps.push(step);
        }
        
        Ok(steps)
    }
    
    /// Generate final answer dari steps
    fn generate_final_answer(&self, steps: &[super::core::ReasoningStep]) -> String {
        if steps.is_empty() {
            return "No answer generated".to_string();
        }
        
        // Simple final answer generation
        format!("Final answer based on {} reasoning steps", steps.len())
    }
    
    /// Sampling dengan temperature dan top-p
    fn _sample_with_temperature(&self, logits: &[f32]) -> Result<usize> {
        // Apply temperature
        let scaled_logits: Vec<f32> = logits.iter()
            .map(|&x| x / self.config.temperature)
            .collect();
        
        // Apply softmax
        let exp_logits: Vec<f32> = scaled_logits.iter()
            .map(|&x| x.exp())
            .collect();
        
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter()
            .map(|&x| x / sum_exp)
            .collect();
        
        // Top-p filtering
        let mut indexed_probs: Vec<(usize, f32)> = probs.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut cumulative_prob = 0.0;
        let mut filtered_indices = Vec::with_capacity(indexed_probs.len());
        
        for (idx, prob) in indexed_probs {
            cumulative_prob += prob;
            filtered_indices.push(idx);
            if cumulative_prob >= self.config.top_p {
                break;
            }
        }
        
        // Sample from filtered indices
        if filtered_indices.is_empty() {
            return Ok(0);
        }
        
        let filtered_probs: Vec<f32> = filtered_indices.iter()
            .map(|&idx| probs[idx])
            .collect();
        
        let filtered_sum: f32 = filtered_probs.iter().sum();
        let normalized_probs: Vec<f32> = filtered_probs.iter()
            .map(|&p| p / filtered_sum)
            .collect();
        
        // Simple sampling
        let mut rand_val = rand::random::<f32>();
        for (i, &prob) in normalized_probs.iter().enumerate() {
            rand_val -= prob;
            if rand_val <= 0.0 {
                return Ok(filtered_indices[i]);
            }
        }
        
        Ok(filtered_indices[filtered_indices.len() - 1])
    }
}

/// SPIN Evaluator untuk mengevaluasi kualitas traces
pub struct SpinEvaluator {
    _config: SpinConfig,
}

impl SpinEvaluator {
    pub fn new(config: SpinConfig) -> Self {
        Self { _config: config }
    }
    
    /// Evaluasi reasoning trace
    pub fn evaluate_trace(&self, trace: &ReasoningTrace) -> Result<f32> {
        let mut score = 0.0;
        
        // Evaluate step quality
        let step_score = self.evaluate_steps(&trace.steps)?;
        score += step_score * 0.7;
        
        // Evaluate final answer
        let answer_score = self.evaluate_final_answer(&trace.final_answer)?;
        score += answer_score * 0.3;
        
        Ok(score)
    }
    
    /// Evaluasi kualitas steps
    fn evaluate_steps(&self, steps: &[super::core::ReasoningStep]) -> Result<f32> {
        if steps.is_empty() {
            return Ok(0.0);
        }
        
        let mut total_score = 0.0;
        
        for step in steps {
            let step_score = self.evaluate_single_step(step)?;
            total_score += step_score;
        }
        
        Ok(total_score / steps.len() as f32)
    }
    
    /// Evaluasi single step
    fn evaluate_single_step(&self, step: &super::core::ReasoningStep) -> Result<f32> {
        let mut score = 0.5; // Base score
        
        // Check step length
        if step.content.len() > 10 && step.content.len() < 500 {
            score += 0.2;
        }
        
        // Check for reasoning keywords
        let reasoning_keywords = ["because", "therefore", "since", "thus", "hence"];
        for keyword in &reasoning_keywords {
            if step.content.to_lowercase().contains(keyword) {
                score += 0.1;
            }
        }
        
        // Check step number consistency
        if step.step_number > 0 {
            score += 0.1;
        }
        
        Ok((score as f32).min(1.0))
    }
    
    /// Evaluasi final answer
    fn evaluate_final_answer(&self, answer: &str) -> Result<f32> {
        let mut score = 0.5;
        
        // Check answer length
        if answer.len() > 5 && answer.len() < 200 {
            score += 0.2;
        }
        
        // Check for answer indicators
        let answer_indicators = ["answer", "result", "conclusion", "therefore"];
        for indicator in &answer_indicators {
            if answer.to_lowercase().contains(indicator) {
                score += 0.15;
            }
        }
        
        // Check for confidence
        if answer.to_lowercase().contains("definitely") || 
           answer.to_lowercase().contains("certainly") {
            score += 0.1;
        }
        
        Ok((score as f32).min(1.0))
    }
    
    /// Bandingkan dua traces
    pub fn compare_traces(&self, trace1: &ReasoningTrace, trace2: &ReasoningTrace) -> Result<GameWinner> {
        let score1 = self.evaluate_trace(trace1)?;
        let score2 = self.evaluate_trace(trace2)?;
        
        let diff = (score1 - score2).abs();
        
        if diff < 0.1 {
            Ok(GameWinner::Draw)
        } else if score1 > score2 {
            Ok(GameWinner::Student)
        } else {
            Ok(GameWinner::Teacher)
        }
    }
}

/// SPIN Trainer
pub struct SpinTrainer {
    config: SpinConfig,
    generator: SpinGenerator,
    evaluator: SpinEvaluator,
    student_model: PolicyModel,
    teacher_model: PolicyModel,
}

impl SpinTrainer {
    pub fn new(
        config: SpinConfig,
        student_model: PolicyModel,
        teacher_model: PolicyModel,
    ) -> Self {
        Self {
            generator: SpinGenerator::new(config.clone()),
            evaluator: SpinEvaluator::new(config.clone()),
            config,
            student_model,
            teacher_model,
        }
    }
    
    /// Jalankan self-play tournament
    pub fn run_tournament(&mut self, prompts: &[String]) -> Result<SelfPlayTournament> {
        let mut games = Vec::with_capacity(prompts.len() * self.config.rounds_per_iteration);
        let mut student_wins = 0;
        let mut teacher_wins = 0;
        let mut draws = 0;
        
        for prompt in prompts {
            for round in 1..=self.config.rounds_per_iteration {
                let game = self.play_game(prompt, round)?;
                
                match game.winner {
                    GameWinner::Student => student_wins += 1,
                    GameWinner::Teacher => teacher_wins += 1,
                    GameWinner::Draw => draws += 1,
                }
                
                games.push(game);
            }
        }
        
        let total_games = games.len();
        let student_win_rate = student_wins as f32 / total_games.max(1) as f32;
        let improvement_score = self.calculate_improvement_score(&games)?;
        
        Ok(SelfPlayTournament {
            id: Uuid::new_v4(),
            games,
            student_wins,
            teacher_wins,
            draws,
            student_win_rate,
            improvement_score,
            created_at: chrono::Utc::now(),
        })
    }
    
    /// Mainkan satu game self-play
    fn play_game(&mut self, prompt: &str, round_number: usize) -> Result<SelfPlayGame> {
        // Generate traces
        let student_trace = self.generator.generate_trace(&self.student_model, prompt)?;
        let teacher_trace = self.generator.generate_trace(&self.teacher_model, prompt)?;
        
        // Evaluate traces
        let student_score = self.evaluator.evaluate_trace(&student_trace)?;
        let teacher_score = self.evaluator.evaluate_trace(&teacher_trace)?;
        
        // Determine winner
        let winner = self.evaluator.compare_traces(&student_trace, &teacher_trace)?;
        
        Ok(SelfPlayGame {
            id: Uuid::new_v4(),
            prompt: prompt.to_string(),
            student_trace,
            teacher_trace,
            round_number,
            student_score,
            teacher_score,
            winner,
            created_at: chrono::Utc::now(),
        })
    }
    
    /// Update models berdasarkan tournament results
    pub fn update_models(&mut self, tournament: &SelfPlayTournament) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut game_count = 0;
        
        for game in &tournament.games {
            let loss = self.calculate_game_loss(game)?;
            total_loss += loss;
            game_count += 1;
            
            // Update student model
            self.update_student_model(game)?;
        }
        
        Ok(total_loss / game_count.max(1) as f32)
    }
    
    /// Promote student to teacher jika improvement signifikan
    pub fn promote_student_to_teacher(&mut self, tournament: &SelfPlayTournament) -> Result<bool> {
        if tournament.improvement_score >= self.config.improvement_threshold {
            // Student becomes new teacher
            self.teacher_model = self.student_model.clone();
            self.teacher_model.set_as_teacher();
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// Get training statistics
    pub fn get_training_stats(&self, tournament: &SelfPlayTournament) -> TrainingStats {
        let avg_student_score = tournament.games.iter()
            .map(|g| g.student_score)
            .sum::<f32>() / tournament.games.len().max(1) as f32;
        
        let avg_teacher_score = tournament.games.iter()
            .map(|g| g.teacher_score)
            .sum::<f32>() / tournament.games.len().max(1) as f32;
        
        TrainingStats {
            total_games: tournament.games.len(),
            student_win_rate: tournament.student_win_rate,
            improvement_score: tournament.improvement_score,
            avg_student_score,
            avg_teacher_score,
            convergence_rate: self.calculate_convergence_rate(tournament),
        }
    }
    
    // Helper methods
    fn calculate_game_loss(&self, game: &SelfPlayGame) -> Result<f32> {
        let score_diff = game.student_score - game.teacher_score;
        
        let loss = match game.winner {
            GameWinner::Student => -score_diff, // Student won, minimize negative loss
            GameWinner::Teacher => score_diff,  // Teacher won, minimize positive loss
            GameWinner::Draw => score_diff.abs(), // Draw, minimize absolute difference
        };
        
        Ok(loss)
    }
    
    fn update_student_model(&mut self, game: &SelfPlayGame) -> Result<()> {
        // Simplified model update based on game outcome
        // In real implementation, this would update neural network parameters
        
        let learning_rate = 0.01;
        let gradient = match game.winner {
            GameWinner::Student => -learning_rate * (1.0 - game.student_score),
            GameWinner::Teacher => learning_rate * game.student_score,
            GameWinner::Draw => learning_rate * (game.student_score - 0.5),
        };
        
        // Apply gradient (simplified)
        Ok(())
    }
    
    fn calculate_improvement_score(&self, games: &[SelfPlayGame]) -> Result<f32> {
        if games.is_empty() {
            return Ok(0.0);
        }
        
        let student_scores: Vec<f32> = games.iter()
            .map(|g| g.student_score)
            .collect();
        
        let teacher_scores: Vec<f32> = games.iter()
            .map(|g| g.teacher_score)
            .collect();
        
        let avg_student = student_scores.iter().sum::<f32>() / student_scores.len() as f32;
        let avg_teacher = teacher_scores.iter().sum::<f32>() / teacher_scores.len() as f32;
        
        Ok((avg_student - avg_teacher).max(0.0))
    }
    
    fn calculate_convergence_rate(&self, tournament: &SelfPlayTournament) -> f32 {
        if tournament.games.len() < 2 {
            return 0.0;
        }
        
        let scores: Vec<f32> = tournament.games.iter()
            .map(|g| g.student_score)
            .collect();
        
        // Calculate variance as convergence indicator
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f32>() / scores.len() as f32;
        
        // Lower variance = higher convergence
        1.0 / (1.0 + variance)
    }
}

/// Training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    pub total_games: usize,
    pub student_win_rate: f32,
    pub improvement_score: f32,
    pub avg_student_score: f32,
    pub avg_teacher_score: f32,
    pub convergence_rate: f32,
}

/// Utility functions
pub mod utils {
    use super::*;
    
    /// Create default SPIN trainer
    pub fn create_default_trainer(
        student_model: PolicyModel,
        teacher_model: PolicyModel,
    ) -> SpinTrainer {
        SpinTrainer::new(
            SpinConfig::default(),
            student_model,
            teacher_model,
        )
    }
    
    /// Generate sample prompts for self-play
    pub fn generate_sample_prompts() -> Vec<String> {
        vec![
            "Solve this math problem: 2 + 2 = ?".to_string(),
            "Explain the concept of gravity".to_string(),
            "Write a short story about a robot".to_string(),
            "What is the capital of France?".to_string(),
            "How do you make a sandwich?".to_string(),
        ]
    }
    
    /// Analyze tournament results
    pub fn analyze_tournament(tournament: &SelfPlayTournament) -> Result<TournamentAnalysis> {
        let games_by_round = tournament.games.iter()
            .fold(HashMap::new(), |mut acc, game| {
                acc.entry(game.round_number).or_insert_with(Vec::new).push(game);
                acc
            });
        
        let mut round_stats = HashMap::new();
        for (round, games) in games_by_round {
            let student_wins = games.iter().filter(|g| matches!(g.winner, GameWinner::Student)).count();
            let round_win_rate = student_wins as f32 / games.len() as f32;
            round_stats.insert(round, round_win_rate);
        }
        
        Ok(TournamentAnalysis {
            total_games: tournament.games.len(),
            student_win_rate: tournament.student_win_rate,
            improvement_score: tournament.improvement_score,
            round_statistics: round_stats.clone(),
            trend: calculate_trend(&round_stats),
        })
    }
    
    fn calculate_trend(round_stats: &HashMap<usize, f32>) -> Trend {
        if round_stats.len() < 2 {
            return Trend::Stable;
        }
        
        let mut rounds: Vec<_> = round_stats.keys().cloned().collect();
        rounds.sort();
        
        let first_rate = round_stats[&rounds[0]];
        let last_rate = round_stats[&rounds[rounds.len() - 1]];
        
        let diff = last_rate - first_rate;
        
        if diff > 0.1 {
            Trend::Improving
        } else if diff < -0.1 {
            Trend::Declining
        } else {
            Trend::Stable
        }
    }
    
    /// Tournament analysis results
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TournamentAnalysis {
        pub total_games: usize,
        pub student_win_rate: f32,
        pub improvement_score: f32,
        pub round_statistics: HashMap<usize, f32>,
        pub trend: Trend,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Trend {
        Improving,
        Declining,
        Stable,
    }
}
