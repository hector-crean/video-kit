use geojson::{FeatureCollection, Feature, Geometry, Value};
use crate::{
    error::Result, 
    error::MaskError, 
    types::{ComputedOutline, ComplexShape},
    typed_geojson::{TypedGeoJson, TypedFeature, TypedFeatureCollection, MaskOutlineProperties, MaskGeoJson}
};







impl ComputedOutline {
    /// Export to typed GeoJSON format
    pub fn to_typed_geojson(&self) -> Result<MaskGeoJson> {
        let mut features = Vec::new();
        
        for (i, shape) in self.shapes.iter().enumerate() {
            let coordinates = if shape.holes.is_empty() {
                // Simple polygon - just exterior ring
                vec![shape.exterior.iter().map(|&[x, y]| vec![x as f64, y as f64]).collect()]
            } else {
                // Polygon with holes - exterior first, then holes
                let mut coords = vec![shape.exterior.iter().map(|&[x, y]| vec![x as f64, y as f64]).collect()];
                for hole in &shape.holes {
                    coords.push(hole.iter().map(|&[x, y]| vec![x as f64, y as f64]).collect());
                }
                coords
            };

            let geometry = Geometry::new(Value::Polygon(coordinates));
            
            let properties = MaskOutlineProperties {
                id: i as u32,
                area: shape.area() as f64,
                has_holes: shape.has_holes(),
                hole_count: shape.holes.len(),
                perimeter: shape.perimeter() as f64,
            };
            
            let typed_feature = TypedFeature::new(Some(geometry), properties);
            features.push(typed_feature);
        }
        
        // Add metadata to foreign members
        let mut foreign_members = serde_json::Map::new();
        foreign_members.insert("image_width".to_string(), serde_json::Value::Number(serde_json::Number::from(self.image_width)));
        foreign_members.insert("image_height".to_string(), serde_json::Value::Number(serde_json::Number::from(self.image_height)));
        foreign_members.insert("shape_count".to_string(), serde_json::Value::Number(serde_json::Number::from(self.shapes.len())));
        
        let typed_collection = TypedFeatureCollection {
            bbox: None,
            features,
            foreign_members: Some(foreign_members),
        };
        
        Ok(TypedGeoJson::FeatureCollection(typed_collection))
    }

    pub fn to_geojson(&self) -> Result<FeatureCollection> {
        let mut features = Vec::new();
        
        for (i, shape) in self.shapes.iter().enumerate() {
            let coordinates = if shape.holes.is_empty() {
                // Simple polygon - just exterior ring
                vec![shape.exterior.iter().map(|&[x, y]| vec![x as f64, y as f64]).collect()]
            } else {
                // Polygon with holes - exterior first, then holes
                let mut coords = vec![shape.exterior.iter().map(|&[x, y]| vec![x as f64, y as f64]).collect()];
                for hole in &shape.holes {
                    coords.push(hole.iter().map(|&[x, y]| vec![x as f64, y as f64]).collect());
                }
                coords
            };

            let geometry = Geometry::new(Value::Polygon(coordinates));
            
            let mut properties = serde_json::Map::new();
            properties.insert("id".to_string(), serde_json::Value::Number(serde_json::Number::from(i)));
            properties.insert("area".to_string(), serde_json::Value::Number(
                serde_json::Number::from_f64(shape.area() as f64).unwrap_or(serde_json::Number::from(0))
            ));
            properties.insert("has_holes".to_string(), serde_json::Value::Bool(shape.has_holes()));
            properties.insert("hole_count".to_string(), serde_json::Value::Number(serde_json::Number::from(shape.holes.len())));
            properties.insert("perimeter".to_string(), serde_json::Value::Number(
                serde_json::Number::from_f64(shape.perimeter() as f64).unwrap_or(serde_json::Number::from(0))
            ));
            
            let feature = Feature {
                bbox: None,
                geometry: Some(geometry),
                id: Some(geojson::feature::Id::Number(serde_json::Number::from(i))),
                properties: Some(properties),
                foreign_members: None,
            };
            
            features.push(feature);
        }
        
        // Add metadata to foreign members of the FeatureCollection
        let mut foreign_members = serde_json::Map::new();
        foreign_members.insert("image_width".to_string(), serde_json::Value::Number(serde_json::Number::from(self.image_width)));
        foreign_members.insert("image_height".to_string(), serde_json::Value::Number(serde_json::Number::from(self.image_height)));
        foreign_members.insert("shape_count".to_string(), serde_json::Value::Number(serde_json::Number::from(self.shapes.len())));
        
        Ok(FeatureCollection {
            bbox: None,
            features,
            foreign_members: Some(foreign_members),
        })
    }
    
    /// Export to GeoJSON and serialize to JSON string
    pub fn to_geojson_string(&self) -> Result<String> {
        let geojson = self.to_geojson()?;
        Ok(serde_json::to_string_pretty(&geojson)?)
    }
    
    /// Save GeoJSON to file
    pub fn save_geojson(&self, path: &str) -> Result<()> {
        let geojson_string = self.to_geojson_string()?;
        std::fs::write(path, geojson_string)?;
        Ok(())
    }
    
    /// Load ComputedOutline from GeoJSON file
    pub fn from_geojson_file(path: &str) -> Result<Self> {
        let geojson_str = std::fs::read_to_string(path)?;
        Self::from_geojson_string(&geojson_str)
    }
    
    /// Load ComputedOutline from GeoJSON string
    pub fn from_geojson_string(geojson_str: &str) -> Result<Self> {
        let geojson: FeatureCollection = geojson_str.parse()?;
        
        // Extract metadata from foreign members
        let foreign_members = geojson.foreign_members.as_ref()
            .ok_or_else(|| MaskError::GeometricComputation("Missing metadata in GeoJSON".to_string()))?;
        
        let image_width = foreign_members.get("image_width")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or_else(|| MaskError::GeometricComputation("Missing or invalid image_width".to_string()))?;
        
        let image_height = foreign_members.get("image_height")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or_else(|| MaskError::GeometricComputation("Missing or invalid image_height".to_string()))?;
        
        let mut shapes = Vec::new();
        
        for feature in geojson.features {
            if let Some(geometry) = feature.geometry {
                if let Value::Polygon(coords) = geometry.value {
                    if coords.is_empty() {
                        continue;
                    }
                    
                    // First ring is exterior
                    let exterior: Vec<[f32; 2]> = coords[0].iter()
                        .map(|coord| [coord[0] as f32, coord[1] as f32])
                        .collect();
                    
                    // Additional rings are holes
                    let holes: Vec<Vec<[f32; 2]>> = coords[1..].iter()
                        .map(|hole| hole.iter()
                            .map(|coord| [coord[0] as f32, coord[1] as f32])
                            .collect())
                        .collect();
                    
                    shapes.push(ComplexShape { exterior, holes });
                }
            }
        }
        
        Ok(ComputedOutline {
            shapes,
            image_width,
            image_height,
        })
    }
} 