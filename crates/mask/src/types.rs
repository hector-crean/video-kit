use serde::{Deserialize, Serialize};
use geo_types::{Coord, LineString, Polygon};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputedOutline {
    /// Multiple separate shapes found in the mask
    pub shapes: Vec<ComplexShape>,
    /// Original image dimensions
    pub image_width: u32,
    pub image_height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexShape {
    /// The exterior boundary of the shape (outer contour)
    pub exterior: Vec<[f32; 2]>,
    /// Interior boundaries (holes within the shape)
    pub holes: Vec<Vec<[f32; 2]>>,
}

impl ComplexShape {
    /// Convert to geo-types Polygon for geometric operations
    pub fn to_geo_polygon(&self) -> Polygon<f32> {
        let exterior_coords: Vec<Coord<f32>> = self.exterior
            .iter()
            .map(|&[x, y]| Coord { x, y })
            .collect();
        
        let exterior_linestring = LineString::new(exterior_coords);
        
        let holes: Vec<LineString<f32>> = self.holes
            .iter()
            .map(|hole| {
                let hole_coords: Vec<Coord<f32>> = hole
                    .iter()
                    .map(|&[x, y]| Coord { x, y })
                    .collect();
                LineString::new(hole_coords)
            })
            .collect();
        
        Polygon::new(exterior_linestring, holes)
    }
    
    /// Calculate the area of the shape (exterior minus holes)
    pub fn area(&self) -> f32 {
        use geo::Area;
        self.to_geo_polygon().unsigned_area()
    }
    
    /// Check if this shape contains holes
    pub fn has_holes(&self) -> bool {
        !self.holes.is_empty()
    }
    
    /// Get the bounding box of the shape
    pub fn bounding_box(&self) -> ([f32; 2], [f32; 2]) {
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        
        for &[x, y] in &self.exterior {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        
        ([min_x, min_y], [max_x, max_y])
    }
    
    /// Get the centroid of the shape
    pub fn centroid(&self) -> [f32; 2] {
        use geo::Centroid;
        if let Some(centroid) = self.to_geo_polygon().centroid() {
            [centroid.x(), centroid.y()]
        } else {
            // Fallback to bounding box center
            let (min, max) = self.bounding_box();
            [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0]
        }
    }
    
    /// Get the perimeter length of the shape (including holes)
    pub fn perimeter(&self) -> f32 {
        // Calculate perimeter by summing the lengths of all rings
        let mut total_perimeter = 0.0;
        
        // Add exterior ring perimeter
        for window in self.exterior.windows(2) {
            let dx = window[1][0] - window[0][0];
            let dy = window[1][1] - window[0][1];
            total_perimeter += (dx * dx + dy * dy).sqrt();
        }
        
        // Add interior rings (holes) perimeter
        for hole in &self.holes {
            for window in hole.windows(2) {
                let dx = window[1][0] - window[0][0];
                let dy = window[1][1] - window[0][1];
                total_perimeter += (dx * dx + dy * dy).sqrt();
            }
        }
        
        total_perimeter
    }
} 