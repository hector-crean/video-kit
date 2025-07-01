use geo_types::{Coord, LineString};
use crate::{error::Result, traits::{ShapeSimplifier, ShapePostProcessor}, types::ComplexShape};

/// Douglas-Peucker simplifier using geo crate's implementation
#[derive(Debug, Clone, Default)]
pub struct DouglasPeuckerSimplifier;

impl ShapeSimplifier for DouglasPeuckerSimplifier {
    fn simplify(&self, shapes: &mut [ComplexShape], tolerance: f32) -> Result<()> {
        use geo::Simplify;
        
        for shape in shapes {
            // Simplify exterior using geo's optimized implementation
            let exterior_coords: Vec<Coord<f32>> = shape.exterior
                .iter()
                .map(|&[x, y]| Coord { x, y })
                .collect();
            let exterior_linestring = LineString::new(exterior_coords);
            let simplified_exterior = exterior_linestring.simplify(&tolerance);
            shape.exterior = simplified_exterior.coords()
                .map(|coord| [coord.x, coord.y])
                .collect();
            
            // Simplify holes using geo's optimized implementation
            for hole in &mut shape.holes {
                let hole_coords: Vec<Coord<f32>> = hole
                    .iter()
                    .map(|&[x, y]| Coord { x, y })
                    .collect();
                let hole_linestring = LineString::new(hole_coords);
                let simplified_hole = hole_linestring.simplify(&tolerance);
                *hole = simplified_hole.coords()
                    .map(|coord| [coord.x, coord.y])
                    .collect();
            }
        }
        
        Ok(())
    }
}

/// Visvalingam-Whyatt simplifier using geo crate's implementation
#[derive(Debug, Clone, Default)]
pub struct VisvalingamWhyattSimplifier;

impl ShapeSimplifier for VisvalingamWhyattSimplifier {
    fn simplify(&self, shapes: &mut [ComplexShape], tolerance: f32) -> Result<()> {
        use geo::SimplifyVw;
        
        for shape in shapes {
            // Simplify exterior using Visvalingam-Whyatt algorithm
            let exterior_coords: Vec<Coord<f32>> = shape.exterior
                .iter()
                .map(|&[x, y]| Coord { x, y })
                .collect();
            let exterior_linestring = LineString::new(exterior_coords);
            let simplified_exterior = exterior_linestring.simplify_vw(&tolerance);
            shape.exterior = simplified_exterior.coords()
                .map(|coord| [coord.x, coord.y])
                .collect();
            
            // Simplify holes using Visvalingam-Whyatt algorithm
            for hole in &mut shape.holes {
                let hole_coords: Vec<Coord<f32>> = hole
                    .iter()
                    .map(|&[x, y]| Coord { x, y })
                    .collect();
                let hole_linestring = LineString::new(hole_coords);
                let simplified_hole = hole_linestring.simplify_vw(&tolerance);
                *hole = simplified_hole.coords()
                    .map(|coord| [coord.x, coord.y])
                    .collect();
            }
        }
        
        Ok(())
    }
}

/// Point decimation simplifier (removes every nth point)
#[derive(Debug, Clone)]
pub struct DecimationSimplifier {
    pub step: usize,
}

impl Default for DecimationSimplifier {
    fn default() -> Self {
        Self { step: 2 }
    }
}

impl ShapeSimplifier for DecimationSimplifier {
    fn simplify(&self, shapes: &mut [ComplexShape], _tolerance: f32) -> Result<()> {
        for shape in shapes {
            // Simplify exterior by keeping every nth point
            if !shape.exterior.is_empty() {
                shape.exterior = shape.exterior
                    .iter()
                    .step_by(self.step)
                    .copied()
                    .collect();
                
                // Ensure we have at least 3 points for a valid polygon
                if shape.exterior.len() < 3 && !shape.exterior.is_empty() {
                    shape.exterior.clear(); // Mark as invalid
                }
            }
            
            // Simplify holes
            for hole in &mut shape.holes {
                if !hole.is_empty() {
                    *hole = hole
                        .iter()
                        .step_by(self.step)
                        .copied()
                        .collect();
                    
                    // Ensure we have at least 3 points for a valid polygon
                    if hole.len() < 3 {
                        hole.clear(); // Mark as invalid
                    }
                }
            }
            
            // Remove invalid holes
            shape.holes.retain(|hole| hole.len() >= 3);
        }
        
        Ok(())
    }
}

/// Minimum area filter using geo crate's area calculation
#[derive(Debug, Clone)]
pub struct MinimumAreaFilter {
    pub min_area: f32,
}

impl Default for MinimumAreaFilter {
    fn default() -> Self {
        Self { min_area: 10.0 }
    }
}

impl ShapePostProcessor for MinimumAreaFilter {
    fn process(&self, shapes: &mut [ComplexShape]) -> Result<()> {
        use geo::Area;
        
        for shape in shapes {
            let polygon = shape.to_geo_polygon();
            if polygon.unsigned_area() < self.min_area {
                // Mark shape as invalid by clearing its points
                shape.exterior.clear();
                shape.holes.clear();
            }
        }
        Ok(())
    }
}

/// Chaikin smoothing post-processor using geo crate's implementation
#[derive(Debug, Clone)]
pub struct ChaikinSmoothingProcessor {
    pub iterations: usize,
}

impl Default for ChaikinSmoothingProcessor {
    fn default() -> Self {
        Self { iterations: 1 }
    }
}

impl ShapePostProcessor for ChaikinSmoothingProcessor {
    fn process(&self, shapes: &mut [ComplexShape]) -> Result<()> {
        use geo::ChaikinSmoothing;
        
        for shape in shapes {
            // Smooth exterior using Chaikin's algorithm
            if !shape.exterior.is_empty() {
                let exterior_coords: Vec<Coord<f32>> = shape.exterior
                    .iter()
                    .map(|&[x, y]| Coord { x, y })
                    .collect();
                let mut exterior_linestring = LineString::new(exterior_coords);
                
                exterior_linestring = exterior_linestring.chaikin_smoothing(self.iterations);
                
                shape.exterior = exterior_linestring.coords()
                    .map(|coord| [coord.x, coord.y])
                    .collect();
            }
            
            // Smooth holes using Chaikin's algorithm
            for hole in &mut shape.holes {
                if !hole.is_empty() {
                    let hole_coords: Vec<Coord<f32>> = hole
                        .iter()
                        .map(|&[x, y]| Coord { x, y })
                        .collect();
                    let mut hole_linestring = LineString::new(hole_coords);
                    
                    hole_linestring = hole_linestring.chaikin_smoothing(self.iterations);
                    
                    *hole = hole_linestring.coords()
                        .map(|coord| [coord.x, coord.y])
                        .collect();
                }
            }
        }
        
        Ok(())
    }
}

/// Geometry validation processor using basic checks
/// Note: Full OGC validation requires additional dependencies
#[derive(Debug, Clone, Default)]
pub struct GeometryValidator;

impl ShapePostProcessor for GeometryValidator {
    fn process(&self, shapes: &mut [ComplexShape]) -> Result<()> {
        for shape in shapes {
            // Basic validation checks
            if shape.exterior.len() < 3 {
                // Mark invalid shapes by clearing their points
                shape.exterior.clear();
                shape.holes.clear();
                continue;
            }
            
            // Check for degenerate holes
            shape.holes.retain(|hole| hole.len() >= 3);
            
            // Check for NaN or infinite coordinates
            let has_invalid_coords = shape.exterior.iter()
                .chain(shape.holes.iter().flatten())
                .any(|&[x, y]| !x.is_finite() || !y.is_finite());
            
            if has_invalid_coords {
                shape.exterior.clear();
                shape.holes.clear();
            }
        }
        
        Ok(())
    }
}

/// Convex hull processor using geo crate's implementation
#[derive(Debug, Clone, Default)]
pub struct ConvexHullProcessor;

impl ShapePostProcessor for ConvexHullProcessor {
    fn process(&self, shapes: &mut [ComplexShape]) -> Result<()> {
        use geo::ConvexHull;
        
        for shape in shapes {
            if !shape.exterior.is_empty() {
                let coords: Vec<Coord<f32>> = shape.exterior
                    .iter()
                    .map(|&[x, y]| Coord { x, y })
                    .collect();
                
                let linestring = LineString::new(coords);
                let convex_hull = linestring.convex_hull();
                
                // Replace exterior with convex hull
                shape.exterior = convex_hull.exterior().coords()
                    .map(|coord| [coord.x, coord.y])
                    .collect();
                
                // Clear holes since convex hull doesn't have holes
                shape.holes.clear();
            }
        }
        
        Ok(())
    }
}

/// Legacy smoothing processor (kept for backward compatibility)
#[derive(Debug, Clone)]
pub struct SmoothingProcessor {
    pub window_size: usize,
}

impl Default for SmoothingProcessor {
    fn default() -> Self {
        Self { window_size: 3 }
    }
}

impl ShapePostProcessor for SmoothingProcessor {
    fn process(&self, shapes: &mut [ComplexShape]) -> Result<()> {
        for shape in shapes {
            // Simple moving average smoothing for exterior
            if shape.exterior.len() > self.window_size {
                let smoothed = self.smooth_contour(&shape.exterior);
                shape.exterior = smoothed;
            }
            
            // Smooth holes
            for hole in &mut shape.holes {
                if hole.len() > self.window_size {
                    let smoothed = self.smooth_contour(hole);
                    *hole = smoothed;
                }
            }
        }
        Ok(())
    }
}

impl SmoothingProcessor {
    fn smooth_contour(&self, contour: &[[f32; 2]]) -> Vec<[f32; 2]> {
        if contour.len() <= self.window_size {
            return contour.to_vec();
        }
        
        let mut smoothed = Vec::new();
        let half_window = self.window_size / 2;
        
        for i in 0..contour.len() {
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut count = 0;
            
            for j in 0..self.window_size {
                let idx = (i + j + contour.len() - half_window) % contour.len();
                sum_x += contour[idx][0];
                sum_y += contour[idx][1];
                count += 1;
            }
            
            smoothed.push([sum_x / count as f32, sum_y / count as f32]);
        }
        
        smoothed
    }
} 