use geo_types::{Coord, LineString, Polygon};
use crate::{error::Result, traits::HoleDetector, types::ComplexShape};

/// Containment-based hole detector
#[derive(Debug, Clone, Default)]
pub struct ContainmentHoleDetector;

impl HoleDetector for ContainmentHoleDetector {
    fn detect_holes(&self, contours: Vec<Vec<[f32; 2]>>) -> Result<Vec<ComplexShape>> {
        use geo::{Contains, Area};
        
        // Convert contours to polygons for geometric analysis
        let mut polygons: Vec<(Polygon<f32>, Vec<[f32; 2]>)> = contours
            .into_iter()
            .map(|points| {
                let coords: Vec<Coord<f32>> = points
                    .iter()
                    .map(|&[x, y]| Coord { x, y })
                    .collect();
                
                let linestring = LineString::new(coords);
                let polygon = Polygon::new(linestring, vec![]);
                (polygon, points)
            })
            .collect();
        
        // Sort by area (largest first) to process outer contours before inner ones
        polygons.sort_by(|a, b| {
            b.0.unsigned_area().partial_cmp(&a.0.unsigned_area()).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let mut shapes = Vec::new();
        let mut used_indices = std::collections::HashSet::new();
        
        for (i, (outer_polygon, outer_points)) in polygons.iter().enumerate() {
            if used_indices.contains(&i) {
                continue;
            }
            
            used_indices.insert(i);
            let mut holes = Vec::new();
            
            // Find holes (polygons contained within this one)
            for (j, (inner_polygon, inner_points)) in polygons.iter().enumerate() {
                if i == j || used_indices.contains(&j) {
                    continue;
                }
                
                // Check if inner polygon is contained within outer polygon
                if outer_polygon.contains(inner_polygon) {
                    holes.push(inner_points.clone());
                    used_indices.insert(j);
                }
            }
            
            shapes.push(ComplexShape {
                exterior: outer_points.clone(),
                holes,
            });
        }
        
        Ok(shapes)
    }
}

/// Simple hole detector that treats each contour as a separate shape
#[derive(Debug, Clone, Default)]
pub struct NoHoleDetector;

impl HoleDetector for NoHoleDetector {
    fn detect_holes(&self, contours: Vec<Vec<[f32; 2]>>) -> Result<Vec<ComplexShape>> {
        let shapes = contours
            .into_iter()
            .map(|points| ComplexShape {
                exterior: points,
                holes: Vec::new(),
            })
            .collect();
        
        Ok(shapes)
    }
}

/// Area-based hole detector (holes are smaller contours inside larger ones)
#[derive(Debug, Clone)]
pub struct AreaBasedHoleDetector {
    pub min_hole_ratio: f32,
    pub max_hole_ratio: f32,
}

impl Default for AreaBasedHoleDetector {
    fn default() -> Self {
        Self {
            min_hole_ratio: 0.01,  // Hole must be at least 1% of parent area
            max_hole_ratio: 0.8,   // Hole must be at most 80% of parent area
        }
    }
}

impl HoleDetector for AreaBasedHoleDetector {
    fn detect_holes(&self, contours: Vec<Vec<[f32; 2]>>) -> Result<Vec<ComplexShape>> {
        use geo::{Contains, Area};
        
        // First pass: convert to polygons and calculate areas
        let mut polygons_with_area: Vec<(Polygon<f32>, Vec<[f32; 2]>, f32)> = contours
            .into_iter()
            .map(|points| {
                let coords: Vec<Coord<f32>> = points
                    .iter()
                    .map(|&[x, y]| Coord { x, y })
                    .collect();
                
                let linestring = LineString::new(coords);
                let polygon = Polygon::new(linestring, vec![]);
                let area = polygon.unsigned_area();
                (polygon, points, area)
            })
            .collect();
        
        // Sort by area (largest first)
        polygons_with_area.sort_by(|a, b| {
            b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let mut shapes = Vec::new();
        let mut used_indices = std::collections::HashSet::new();
        
        for (i, (outer_polygon, outer_points, outer_area)) in polygons_with_area.iter().enumerate() {
            if used_indices.contains(&i) {
                continue;
            }
            
            used_indices.insert(i);
            let mut holes = Vec::new();
            
            // Find holes based on containment and area ratio
            for (j, (inner_polygon, inner_points, inner_area)) in polygons_with_area.iter().enumerate() {
                if i == j || used_indices.contains(&j) {
                    continue;
                }
                
                let area_ratio = inner_area / outer_area;
                
                // Check if inner polygon is contained within outer polygon and meets size criteria
                if outer_polygon.contains(inner_polygon) 
                    && area_ratio >= self.min_hole_ratio 
                    && area_ratio <= self.max_hole_ratio {
                    holes.push(inner_points.clone());
                    used_indices.insert(j);
                }
            }
            
            shapes.push(ComplexShape {
                exterior: outer_points.clone(),
                holes,
            });
        }
        
        Ok(shapes)
    }
} 